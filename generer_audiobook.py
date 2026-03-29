#!/usr/bin/env python3
"""
Génère le livre audio de "L'Instinct du Pouvoir" via l'API TTS de Mistral (Voxtral).

Usage:
    set MISTRAL_API_KEY=...
    python generer_audiobook.py [-n NB_CHAPITRES]
"""

import argparse
import base64
import os
import re
import time
from pathlib import Path

import httpx

# ── Configuration ──────────────────────────────────────────────

VOICE = "fr_marie_excited"
MODEL = "voxtral-mini-tts-latest"
MAX_CHARS = 2500
MISTRAL_API_URL = "https://api.mistral.ai/v1/audio/speech"

# Durées de silence en secondes
SILENCE_BETWEEN_CHAPTERS = 2.5
SILENCE_BEFORE_H1 = 2.0
SILENCE_BEFORE_H2 = 1.5
SILENCE_BEFORE_H3 = 1.0

MACHIAVEL_DIR = Path(__file__).parent
OUTPUT_DIR = MACHIAVEL_DIR / "audiobook"
CHAPTERS_DIR = OUTPUT_DIR / "chapitres"
CHUNKS_DIR = OUTPUT_DIR / "chunks"

CHAPTERS = [
    ("Avant_Propos.md",                        "00_Avant_Propos"),
    ("Introduction.md",                         "00b_Introduction"),
    ("Principe_01_Caresser_Ecraser.md",         "01_Caresser_Ecraser"),
    ("Principe_02_Craint_Aime.md",              "02_Craint_Aime"),
    ("Principe_03_Paraitre_Etre.md",            "03_Paraitre_Etre"),
    ("Principe_04_Renard_Lion.md",              "04_Renard_Lion"),
    ("Principe_05_Qui_Rend_Puissant.md",        "05_Qui_Rend_Puissant"),
    ("Principe_06_Cruautes_Bien_Employees.md",  "06_Cruautes_Bien_Employees"),
    ("Principe_07_Deleguer_Odieux.md",          "07_Deleguer_Odieux"),
    ("Principe_08_Meilleure_Forteresse.md",     "08_Meilleure_Forteresse"),
    ("Principe_09_Prevoir_Maux.md",             "09_Prevoir_Maux"),
    ("Principe_10_Jamais_Neutre.md",            "10_Jamais_Neutre"),
    ("Principe_11_Fortune_Moitie.md",           "11_Fortune_Moitie"),
    ("Principe_12_Prophetes_Armes.md",          "12_Prophetes_Armes"),
    ("Principe_13_Reforme_Brutale.md",          "13_Reforme_Brutale"),
    ("Principe_14_Conseil_Adulateur.md",        "14_Conseil_Adulateur"),
    ("Principe_15_Agir_Soi_Meme.md",            "15_Agir_Soi_Meme"),
    ("Principe_16_Reputation_Percue.md",        "16_Reputation_Percue"),
    ("Principe_17_Anciens_Nouveaux.md",         "17_Anciens_Nouveaux"),
    ("Principe_18_Occasions_Crises.md",         "18_Occasions_Crises"),
    ("Principe_19_Gloire_Actions.md",           "19_Gloire_Actions"),
    ("Principe_20_Nouveaux_Anciens.md",         "20_Nouveaux_Anciens"),
    ("Principe_21_Unis_Divises.md",             "21_Unis_Divises"),
    ("Principe_22_Virtus_Fortuna.md",           "22_Virtus_Fortuna"),
    ("Principe_23_Liberte_Necessaire.md",       "23_Liberte_Necessaire"),
    ("Principe_24_Verite_Effective.md",         "24_Verite_Effective"),
    ("Conclusion.md",                           "25_Conclusion"),
]


# ── Nettoyage du Markdown ──────────────────────────────────────

def clean_markdown(text: str) -> str:
    # Supprimer la section des notes de bas de page
    parts = re.split(r'\n---\s*\n', text)
    if len(parts) > 1 and re.search(r'\[\^\d+\]:', parts[-1]):
        text = '\n---\n'.join(parts[:-1])

    # Supprimer les références aux notes [^1], [^2], etc.
    text = re.sub(r'\[\^\d+\]', '', text)

    # Titres avec pauses selon le niveau
    def heading_to_pause(m):
        level = len(m.group(1))
        title = m.group(2)
        if title == title.upper() and len(title) > 2:
            title = title.title()
        return f'\n<<SECTION:{level}>>\n\n' + title + '.\n\n'

    text = re.sub(r'^(#{1,6})\s+(.+)$', heading_to_pause, text, flags=re.MULTILINE)

    # Citations blockquotes
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # Gras et italique
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.+?)_{1,3}', r'\1', text)

    # Liens markdown
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Lignes horizontales
    text = re.sub(r'^-{3,}\s*$', '', text, flags=re.MULTILINE)

    # Sauts de ligne multiples
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def remove_section_markers(text: str) -> tuple[str, int]:
    """Retire les marqueurs <<SECTION:N>> et retourne (texte, niveau_titre)."""
    level = 0
    m = re.match(r'^<<SECTION:(\d)>>', text.strip())
    if m:
        level = int(m.group(1))
    text = re.sub(r'<<SECTION:\d>>', '', text).strip()
    return text, level


# ── Découpage en chunks ────────────────────────────────────────

def split_into_chunks(text: str, max_chars: int = MAX_CHARS) -> list[tuple[str, int]]:
    """Retourne une liste de (texte, niveau_titre)."""
    chunks = []
    remaining = text

    while len(remaining) > max_chars:
        cut_zone = remaining[:max_chars]
        cut_pos = -1

        # Priorité 1 : avant un <<SECTION:N>>
        for match in re.finditer(r'<<SECTION:\d>>', cut_zone):
            if match.start() >= 500:
                cut_pos = match.start()

        # Priorité 2 : entre paragraphes
        if cut_pos == -1:
            for match in re.finditer(r'\n\n', cut_zone):
                if match.start() >= 500:
                    cut_pos = match.start()

        # Priorité 3 : fin de phrase
        if cut_pos == -1:
            for match in re.finditer(r'[.!?…]\s', cut_zone):
                cut_pos = match.end()

        # Dernier recours
        if cut_pos == -1:
            cut_pos = cut_zone.rfind(' ')
            if cut_pos == -1:
                cut_pos = max_chars

        chunk_raw = remaining[:cut_pos]
        text_clean, level = remove_section_markers(chunk_raw)
        if text_clean:
            chunks.append((text_clean, level))
        remaining = remaining[cut_pos:].strip()

    if remaining.strip():
        text_clean, level = remove_section_markers(remaining)
        if text_clean:
            chunks.append((text_clean, level))

    return chunks


# ── Génération TTS ─────────────────────────────────────────────

def generate_tts(api_key: str, text: str, output_path: Path) -> None:
    """Génère un WAV via Mistral TTS."""
    padded_text = text + "\n\n\n\n\n"

    retries = 6
    for attempt in range(retries):
        try:
            resp = httpx.post(
                MISTRAL_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": MODEL,
                    "input": padded_text,
                    "voice": VOICE,
                    "response_format": "wav",
                },
                timeout=60,
            )
            resp.raise_for_status()
            audio_bytes = base64.b64decode(resp.json()["audio_data"])
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            return
        except Exception as e:
            if attempt < retries - 1:
                wait = min(5 * (attempt + 1), 30)
                print(f"\n      Erreur (tentative {attempt+1}/{retries}): Retry dans {wait}s...")
                time.sleep(wait)
            else:
                raise


def get_wav_params(wav_path: Path) -> tuple:
    """Lit les paramètres (channels, sampwidth, framerate) d'un WAV."""
    import wave as wave_mod
    with wave_mod.open(str(wav_path), 'rb') as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate()


def make_silence(channels: int, sampwidth: int, framerate: int, duration_s: float) -> bytes:
    """Génère des frames PCM silencieuses."""
    n_frames = int(framerate * duration_s)
    return b'\x00' * (n_frames * channels * sampwidth)


def concatenate_wav_files(entries: list[tuple[Path, float]], output_path: Path) -> None:
    """Concatène des WAV avec silence optionnel avant chaque fichier.

    entries = liste de (chemin_wav, silence_avant_en_secondes)
    """
    import wave as wave_mod

    channels, sampwidth, framerate = get_wav_params(entries[0][0])

    with wave_mod.open(str(output_path), 'wb') as out_wf:
        out_wf.setnchannels(channels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(framerate)

        for wav_path, silence_before in entries:
            if silence_before > 0:
                out_wf.writeframes(make_silence(channels, sampwidth, framerate, silence_before))
            with wave_mod.open(str(wav_path), 'rb') as in_wf:
                out_wf.writeframes(in_wf.readframes(in_wf.getnframes()))


# ── Pipeline principal ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Génère le livre audio de L'Instinct du Pouvoir")
    parser.add_argument("-n", "--max-chapters", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("ERREUR : Définissez la variable d'environnement MISTRAL_API_KEY")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    CHAPTERS_DIR.mkdir(exist_ok=True)
    CHUNKS_DIR.mkdir(exist_ok=True)

    chapters_to_process = CHAPTERS[:args.max_chapters] if args.max_chapters else CHAPTERS
    chapter_entries: list[tuple[Path, float]] = []

    for md_file, chapter_name in chapters_to_process:
        chapter_output = CHAPTERS_DIR / f"{chapter_name}.wav"

        if chapter_output.exists():
            print(f"  [OK] {chapter_name} existe deja, on passe")
            silence = SILENCE_BETWEEN_CHAPTERS if chapter_entries else 0.0
            chapter_entries.append((chapter_output, silence))
            continue

        print(f"\n{'='*60}")
        print(f"  Chapitre : {chapter_name}")
        print(f"{'='*60}")

        md_path = MACHIAVEL_DIR / md_file
        if not md_path.exists():
            print(f"  ERREUR : {md_file} introuvable, on passe")
            continue

        raw_text = md_path.read_text(encoding="utf-8")
        if chapter_name == "00_Avant_Propos":
            raw_text = "# L'Instinct du Pouvoir\n\n## Vingt-quatre principes de Machiavel\n\n" + raw_text
        clean_text = clean_markdown(raw_text)
        print(f"  Texte nettoyé : {len(clean_text)} caractères")

        chunks = split_into_chunks(clean_text)
        print(f"  Découpé en {len(chunks)} morceaux")

        chunk_entries: list[tuple[Path, float]] = []
        chapter_chunk_dir = CHUNKS_DIR / chapter_name
        chapter_chunk_dir.mkdir(exist_ok=True)

        for i, (chunk_text, section_level) in enumerate(chunks):
            chunk_path = chapter_chunk_dir / f"chunk_{i:03d}.wav"

            silence_map = {1: SILENCE_BEFORE_H1, 2: SILENCE_BEFORE_H2, 3: SILENCE_BEFORE_H3}
            silence_before = silence_map.get(section_level, 0.0) if i > 0 else 0.0

            if chunk_path.exists():
                print(f"    [OK] Chunk {i+1}/{len(chunks)} existe deja")
                chunk_entries.append((chunk_path, silence_before))
                continue

            chunk_text_path = chapter_chunk_dir / f"chunk_{i:03d}.txt"
            if chunk_text_path.exists():
                chunk_text = chunk_text_path.read_text(encoding="utf-8")
            else:
                chunk_text_path.write_text(chunk_text, encoding="utf-8")

            print(f"    > Chunk {i+1}/{len(chunks)} ({len(chunk_text)} chars, silence={silence_before}s)...", end=" ", flush=True)
            try:
                generate_tts(api_key, chunk_text, chunk_path)
                print("OK")
            except Exception as e:
                print(f"\n      ECHEC: {e}")
                raise

            chunk_entries.append((chunk_path, silence_before))

        print(f"  Fusion des chunks en {chapter_name}.wav...")
        concatenate_wav_files(chunk_entries, chapter_output)
        silence = SILENCE_BETWEEN_CHAPTERS if chapter_entries else 0.0
        chapter_entries.append((chapter_output, silence))
        print(f"  [OK] {chapter_name} terminé")

    if chapter_entries:
        final_output = OUTPUT_DIR / "L_Instinct_du_Pouvoir.wav"
        print(f"\n{'='*60}")
        print(f"  Fusion finale : {final_output}")
        print(f"{'='*60}")
        concatenate_wav_files(chapter_entries, final_output)
        size_mb = final_output.stat().st_size / (1024 * 1024)
        print(f"\n  [OK] Livre audio complet : {final_output}")
        print(f"    Taille : {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
