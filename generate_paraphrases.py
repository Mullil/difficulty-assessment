import csv
import os
import re

from vllm import LLM, SamplingParams
from tqdm import tqdm


# CEFR mapping: floats to CEFR strings. (intermediates towards the polar edges)
FLOAT_TO_CEFR = {
    1.0: "A1",
    1.5: "A1+",
    2.0: "A2",
    2.5: "A2+",
    3.0: "B1",
    3.5: "B1+",
    4.0: "B2",
    4.5: "B2+",
    5.0: "C1",
    5.5: "C1+",
    6.0: "C2",
}

# Output label per CEFR string. Floats so half-steps are preserved.
CEFR_TO_LABEL = {
    "A1": 1.0, "A1+": 1.5,
    "A2": 2.0, "A2+": 2.5,
    "B1": 3.0, "B1+": 3.5,
    "B2": 4.0, "B2+": 4.5,
    "C1": 5.0, "C1+": 5.5,
    "C2": 6.0,
}

CEFR_DESCRIPTIONS_FI = {
    "A1":  "alkeistaso – hyvin yksinkertainen sanasto, lyhyet lauseet, perusaikamuodot, vain konkreettiset aiheet",
    "A1+": "alkeistason ja perustason väliltä – pääosin hyvin yksinkertaista sanastoa ja lyhyitä lauseita kuten A1:llä, mutta mukana on jo joitakin sidesanoja (ja, mutta, koska) ja hieman pidempiä lauseita",
    "A2":  "perustaso – yksinkertainen sanasto, lyhyet lauseet, perusaikamuodot, arkiset aiheet",
    "A2+": "perustason ja keskitason väliltä – arkinen sanasto kuten A2:lla, mutta lauseet ovat välillä yhdyslauseita ja mukana on yksinkertaisia mielipiteen ja perustelun ilmauksia",
    "B1":  "keskitaso – yleinen sanasto, kohtuullisen monimutkaiset lauseet, päämuodot tutuissa aiheissa",
    "B1+": "keskitason ja ylemmän keskitason väliltä – yleinen sanasto kuten B1:llä, mutta lauserakenteessa on enemmän vaihtelua, sivulauseita esiintyy useammin ja sanasto on hieman täsmällisempää",
    "B2":  "ylempi keskitaso – monipuolinen sanasto, monimutkaisia lauseita, useita aikamuotoja, abstrakteja aiheita",
    "B2+": "ylemmän keskitason ja edistyneen tason väliltä – monipuolista sanastoa kuten B2:lla, mutta mukana on enemmän abstrakteja käsitteitä, vivahteikkaita ilmauksia ja monimutkaisempaa lauserakennetta",
    "C1":  "edistynyt taso – kehittynyt sanasto, monimutkainen syntaksi, vivahteikas ilmaisu, idiomaattinen kieli",
    "C1+": "edistyneen tason ja taitajatason väliltä – kehittynyttä sanastoa kuten C1:llä, mutta tekstissä esiintyy nominalisaatioita, tiiviimpää lauserakennetta ja paikoittain alakohtaista erikoissanastoa",
    "C2":  "taitajataso – laaja sanasto, hyvin monimutkainen syntaksi, nominalisaatiot, alakohtainen erikoissanasto, tiivis lauserakenne",
}


def map_label_to_cefr(label_value) -> str:
    label = round(float(label_value), 1)
    if label in FLOAT_TO_CEFR:
        return FLOAT_TO_CEFR[label]
    raise ValueError(f"Unexpected label value: {label_value}")


def build_messages(text: str, cefr_level: str) -> list[dict]:
    """Build chat messages. Instruction is in Finnish so the model stays in Finnish."""
    description = CEFR_DESCRIPTIONS_FI[cefr_level]
    system = (
        "Olet suomen kielen asiantuntija ja CEFR-tasojen tuntija. "
        "Tehtäväsi on muotoilla uudelleen annettu suomenkielinen teksti niin, että "
        f"sen CEFR-vaikeustaso pysyy täsmälleen samana: {cefr_level} ({description}). "
        "Sääntöjä:\n"
        "- Kirjoita vain suomeksi.\n"
        "- Muotoile lauseet ja ilmaukset uudelleen – älä toista alkuperäistä sanamuotoa.\n"
        "- Säilytä merkitys ja kaikki olennainen tieto.\n"
        "- Sovita sanasto, lauserakenteet ja kielioppi tarkasti tasolle "
        f"{cefr_level}.\n"
        "- Älä yksinkertaista äläkä vaikeuta tekstiä.\n"
        "- Älä anna käännöksiä, selityksiä, kommentteja tai otsikoita. "
        "Tulosta ainoastaan uudelleen muotoiltu teksti."
    )
    user = f"Uudelleen muotoiltava teksti:\n{text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_existing_ids(output_path: str) -> set[str]:
    """Read IDs already written to the output CSV (for resume)."""
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # header
        except StopIteration:
            return done
        for row in reader:
            if row:
                done.add(row[0])
    return done


# Patterns the model commonly adds; we strip them post-hoc.
_PREAMBLE_PATTERNS = [
    re.compile(
        r"^\s*(uudelleen\s+muotoiltu\s+teksti|tässä\s+on\s+uudelleen\s+muotoiltu\s+teksti)[:\-–—]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(parafraasi|parafraasattu\s+teksti|tässä\s+on\s+parafraasi)[:\-–—]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(here(?:'s| is)?(?: the| a)? paraphrase)[:\-–—]?\s*", re.IGNORECASE
    ),
    re.compile(r"^\s*(translation|english\s+translation|käännös)[:\-–—]?\s*", re.IGNORECASE),
    re.compile(r"^\s*(note|huom|huomautus)[:\-–—]?\s*", re.IGNORECASE),
]

# If any of these markers appear in the output, cut everything from there onward.
_TRAILING_MARKERS = [
    "\nTranslation:",
    "\nEnglish translation:",
    "\nKäännös:",
    "\nNote:",
    "\nHuomautus:",
    "\nHuom:",
    "\nExplanation:",
    "\nSelitys:",
    # Chat-template turn markers — Poro-chat sometimes emits these.
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "<|im_start|>",
    "[INST]",
    "</s>",
]


def clean_output(text: str) -> str:
    """Strip common preamble lines and cut at trailing markers."""
    cleaned = text.strip()

    # Cut at the first trailing marker, if any.
    cut_at = len(cleaned)
    for marker in _TRAILING_MARKERS:
        idx = cleaned.find(marker)
        if idx != -1 and idx < cut_at:
            cut_at = idx
    cleaned = cleaned[:cut_at].strip()

    # Strip preamble lines repeatedly (handles "Tässä on parafraasi:\nParafraasi:...").
    changed = True
    while changed:
        changed = False
        for pat in _PREAMBLE_PATTERNS:
            new = pat.sub("", cleaned, count=1)
            if new != cleaned:
                cleaned = new.strip()
                changed = True

    # Strip wrapping quotes if the whole answer is quoted.
    if len(cleaned) >= 2 and cleaned[0] in "\"'“„«" and cleaned[-1] in "\"'”»":
        cleaned = cleaned[1:-1].strip()

    return cleaned


def augment_csv(
    input_path: str = "train.csv",
    output_path: str = "train_llm_augmented.csv",
    skipped_path: str = "train_llm_augmented.skipped.csv",
    model_name: str = os.environ.get(
        "MODEL_PATH", "LumiOpen/Poro-34B-chat"
    ),
    chunk_size: int = 64,
    max_new_tokens: int = 512,
    max_model_len: int = 2048,
    temperature: float = 0.4,
    top_p: float = 0.9,
    seed: int = 42,
) -> None:
    """Paraphrase every row in `input_path` with vLLM, writing chunked output."""
    # Read input
    with open(input_path, "r", encoding="utf-8", newline="") as f_in:
        rows = list(csv.DictReader(f_in))

    already_done = load_existing_ids(output_path)
    if already_done:
        print(f"Resuming: {len(already_done)} rows already written, skipping them.")

    pending = [r for r in rows if r["#"] not in already_done]
    if not pending:
        print("Nothing to do — all rows already paraphrased.")
        return

    print(f"Loading {model_name}...")
    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        seed=seed,
        tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
        trust_remote_code=True,
    )

    tokenizer = llm.get_tokenizer()

    # Stop sequences: cut generation as soon as the model tries to
    # add commentary/translation/another turn.
    stop_strings = [
        "\nTranslation:",
        "\nEnglish translation:",
        "\nKäännös:",
        "\nNote:",
        "\nHuomautus:",
        "\nHuom:",
        "\nExplanation:",
        "\nSelitys:",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|end_of_text|>",
        "<|im_end|>",
        "<|im_start|>",
        "[INST]",
        "</s>",
    ]

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=stop_strings,
        repetition_penalty=1.05,
    )

    # Reserve room for generated tokens; small safety margin.
    max_prompt_tokens = max_model_len - max_new_tokens - 8
    if max_prompt_tokens <= 0:
        raise ValueError(
            f"max_model_len ({max_model_len}) too small for max_new_tokens "
            f"({max_new_tokens})."
        )

    prepared = []
    skipped = []

    for row in pending:
        try:
            cefr = map_label_to_cefr(row["label"])
        except ValueError as e:
            print(f"Skipping row {row['#']}: {e}")
            skipped.append({"id": row["#"], "reason": f"unmapped_label: {e}"})
            continue

        messages = build_messages(row["text"], cefr)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for tokenizers without a chat template.
            prompt = (
                f"{messages[0]['content']}\n\n"
                f"{messages[1]['content']}\n\n"
                "Uudelleen muotoiltu teksti:\n"
            )

        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > max_prompt_tokens:
            print(
                f"Skipping row {row['#']}: prompt has {len(token_ids)} tokens, "
                f"limit is {max_prompt_tokens}."
            )
            skipped.append({
                "id": row["#"],
                "reason": f"prompt_too_long: {len(token_ids)} > {max_prompt_tokens}",
            })
            continue

        prepared.append({
            "id": row["#"],
            "prompt": prompt,
            "label": CEFR_TO_LABEL[cefr],
        })

    if skipped:
        write_header_skipped = (
            not os.path.exists(skipped_path) or os.path.getsize(skipped_path) == 0
        )
        with open(skipped_path, "a", encoding="utf-8", newline="") as f_skip:
            writer = csv.writer(f_skip)
            if write_header_skipped:
                writer.writerow(["#", "reason"])
            for item in skipped:
                writer.writerow([item["id"], item["reason"]])
        print(f"Logged {len(skipped)} skipped row(s) to {skipped_path}.")

    if not prepared:
        print("Nothing to generate after filtering.")
        return

    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    with open(output_path, "a", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(["#", "text", "label"])
            f_out.flush()

        for start in tqdm(range(0, len(prepared), chunk_size), desc="Chunks"):
            chunk = prepared[start : start + chunk_size]
            prompts = [item["prompt"] for item in chunk]

            try:
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
            except ValueError as e:
                print(f"Chunk failed ({e}); retrying one prompt at a time.")
                outputs = []
                kept_chunk = []
                for item in chunk:
                    try:
                        single = llm.generate([item["prompt"]], sampling, use_tqdm=False)
                        outputs.extend(single)
                        kept_chunk.append(item)
                    except ValueError as e_single:
                        print(f"Skipping row {item['id']}: {e_single}")
                        with open(skipped_path, "a", encoding="utf-8", newline="") as f_skip:
                            csv.writer(f_skip).writerow(
                                [item["id"], f"generate_failed: {e_single}"]
                            )
                chunk = kept_chunk

            for item, out in zip(chunk, outputs):
                paraphrased = clean_output(out.outputs[0].text)
                writer.writerow([item["id"], paraphrased, item["label"]])
            f_out.flush()
            os.fsync(f_out.fileno())


if __name__ == "__main__":
    augment_csv()