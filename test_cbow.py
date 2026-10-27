import numpy as np
import re
from collections import Counter

# ============================================================
# CONFIGURAZIONE
# ============================================================
WINDOW_SIZE = 3   # deve essere uguale a quella usata nel training

# ============================================================
# TOKENIZZAZIONE (identica al training)
# ============================================================
def tokenizza(frase):
    return re.findall(r"[a-zàèéìíòóùú]+", frase.lower())

# ============================================================
# CARICA VOCABOLARIO, PARAMETRI E FRASI
# ============================================================
with open("vocabolario.txt", "r", encoding="utf-8") as f:
    vocabolario = [r.strip() for r in f if r.strip()]

parola2idx = {p: i for i, p in enumerate(vocabolario)}
idx2parola = {i: p for i, p in enumerate(vocabolario)}

embeddings = np.load("embeddings.npy")
W_out      = np.load("W_out.npy")

# Frasi già tokenizzate, salvate dal training
def leggi_frasi(nome_file):
    with open(nome_file, "r", encoding="utf-8") as f:
        return [[parola2idx[p] for p in r.split()] for r in f if r.strip()]

frasi_training = leggi_frasi("frasi_training.txt")
frasi_verifica = leggi_frasi("frasi_verifica.txt")

# ============================================================
# FUNZIONI
# ============================================================
def softmax(scores):
    scores = scores - scores.max()
    exp    = np.exp(scores)
    return exp / exp.sum()

def probabilita(contesto_idx):
    v_medio = embeddings[contesto_idx].mean(axis=0)
    return softmax(W_out.T @ v_medio)

def contesto_attorno(frase, i):
    # le WINDOW_SIZE parole prima e dopo la posizione i (come nel training)
    inizio = max(0, i - WINDOW_SIZE)
    fine   = min(len(frase), i + WINDOW_SIZE + 1)
    return [frase[j] for j in range(inizio, fine) if j != i]

def completa(frase_con_buco, top_n=4):
    # "il gatto ___ il topo" -> contesto = parole attorno al buco
    sinistra, destra = frase_con_buco.split("___")
    sinistra = [parola2idx[p] for p in tokenizza(sinistra) if p in parola2idx]
    destra   = [parola2idx[p] for p in tokenizza(destra)   if p in parola2idx]
    contesto = sinistra[-WINDOW_SIZE:] + destra[:WINDOW_SIZE]
    probs    = probabilita(contesto)
    top      = np.argsort(probs)[::-1][:top_n]
    return [(idx2parola[i], probs[i]) for i in top]

def coseno(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def parole_simili(parola, top_n=4):
    if parola not in parola2idx:
        return []
    idx = parola2idx[parola]
    sim = [(p, coseno(embeddings[idx], embeddings[i]))
           for i, p in enumerate(vocabolario) if i != idx]
    sim.sort(key=lambda x: x[1], reverse=True)
    return sim[:top_n]

# ============================================================
# TEST 1: COMPLETA LA FRASE
# ============================================================
# Il buco "___" può stare in qualsiasi punto: il CBOW guarda
# le parole a sinistra E a destra, proprio come nel training.
print("=" * 60)
print("TEST 1: COMPLETA LA FRASE")
print("=" * 60)

casi = [
    ("il gatto ___ il topo",        "mangia"),
    ("il cane mangia l'___",        "osso"),
    ("la bambina beve il ___",      "latte"),
    ("il cane gioca con la ___",    "palla"),
    ("il gatto dorme sul ___",      "divano"),
    ("il bambino lava i ___",       "denti"),
    ("la bambina legge un ___",     "libro"),
    ("il ___ abbaia",               "cane"),
    ("il ___ miagola",              "gatto"),
    ("il gatto mangia il ___",      "topo"),
    ("La bambina ha ___ a leggere", "imparato"),
    
]

for frase, attesa in casi:
    risultati = completa(frase)
    trovata   = attesa in [p for p, _ in risultati]
    print(f"\n  {frase}   (attesa: {attesa}) {'✓' if trovata else '✗'}")
    for p, prob in risultati:
        print(f"    {p:<15} {prob:.4f}")

# ============================================================
# TEST 2: ACCURATEZZA SU FRASI MAI VISTE
# ============================================================
# Per ogni parola di ogni frase: la nascondo, chiedo al modello
# di indovinarla e controllo se è la prima risposta (top-1)
# o tra le prime 5 (top-5).
# Confronto con un "modello banale" che risponde sempre
# le parole più frequenti del training, senza guardare il contesto.
print(f"\n{'=' * 60}")
print("TEST 2: ACCURATEZZA")
print("=" * 60)

frequenze  = Counter(p for frase in frasi_training for p in frase)
piu_comuni = [p for p, _ in frequenze.most_common(20)]
banale_top1, banale_top5 = piu_comuni[0], set(piu_comuni[:5])

def accuratezza(frasi):
    r = Counter()
    for frase in frasi:
        for i, target in enumerate(frase):
            contesto = contesto_attorno(frase, i)
            if not contesto:
                continue
            top5 = np.argsort(probabilita(contesto))[::-1][:5]
            rara = target not in piu_comuni   # parola "di contenuto"
            r["tot"]       += 1
            r["top1"]      += top5[0] == target
            r["top5"]      += target in top5
            r["b_top1"]    += target == banale_top1
            r["b_top5"]    += target in banale_top5
            r["rare"]      += rara
            r["rare_top1"] += rara and top5[0] == target
            r["rare_top5"] += rara and target in top5
    return r

def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "  n/d"

print(f"\n  Modello banale: risponde sempre '{idx2parola[banale_top1]}' "
      f"(top-5: {', '.join(idx2parola[p] for p in piu_comuni[:5])})")

for nome, frasi in [("training (già viste)", frasi_training),
                    ("verifica (mai viste)", frasi_verifica)]:
    r = accuratezza(frasi)
    print(f"\n  Frasi di {nome}: {r['tot']} parole da indovinare")
    print(f"    {'':<28}{'top-1':>8}{'top-5':>8}")
    print(f"    {'CBOW':<28}{pct(r['top1'], r['tot']):>8}{pct(r['top5'], r['tot']):>8}")
    print(f"    {'modello banale':<28}{pct(r['b_top1'], r['tot']):>8}{pct(r['b_top5'], r['tot']):>8}")
    print(f"    {'CBOW, solo parole rare *':<28}{pct(r['rare_top1'], r['rare']):>8}{pct(r['rare_top5'], r['rare']):>8}")

print("\n  * parole rare = escluse le 20 più frequenti (il, la, e, non, ...).")
print("    Su queste il modello banale farebbe 0%.")

# ============================================================
# TEST 3: PAROLE PIÙ SIMILI
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: PAROLE PIÙ SIMILI")
print("=" * 60)

da_testare = [
    "gatto", "cane", "bambino", "bambina",
    "mangia", "dorme", "corre", "beve"
]

for parola in da_testare:
    print(f"\n  '{parola}' è più simile a:")
    for p, s in parole_simili(parola):
        print(f"    {p:<15} {s:+.4f}")

# ============================================================
# TEST 4: CONFRONTO DIRETTO TRA COPPIE
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: CONFRONTO DIRETTO TRA COPPIE")
print("=" * 60)

coppie = [
    ("gatto",   "cane"),
    ("gatto",   "bambino"),
    ("bambino", "bambina"),
    ("mangia",  "beve"),
    ("mangia",  "abbaia"),
    ("dorme",   "corre"),
    ("gatto",   "mangia"),
    ("cane",    "abbaia"),
]

for a, b in coppie:
    s = coseno(embeddings[parola2idx[a]], embeddings[parola2idx[b]])
    print(f"  '{a}' ↔ '{b}' → {s:+.4f}")
