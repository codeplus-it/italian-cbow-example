import numpy as np
import re
import argparse

# ============================================================
# PARAMETRI DA RIGA DI COMANDO
# ============================================================
# Esempi:
#   python cbow_from_files.py -save_loss y
#       salva la loss di ogni epoca in un csv (per fare il grafico)
#   python cbow_from_files.py -save_loss y -subsampling y
#       come sopra, ma scarta a caso parte delle parole più frequenti
parser = argparse.ArgumentParser()
parser.add_argument("-save_loss", choices=["y", "n"], default="n",
                    help="y = salva la loss di ogni epoca in un file csv")
parser.add_argument("-subsampling", choices=["y", "n"], default="n",
                    help="y = scarta a caso parte delle parole frequenti (come Word2Vec)")
args = parser.parse_args()
SALVA_LOSS  = args.save_loss == "y"
SUBSAMPLING = args.subsampling == "y"

# Nomi diversi così le due prove non si sovrascrivono
FILE_LOSS = "loss_subsampling.csv" if SUBSAMPLING else "loss.csv"

# ============================================================
# TOKENIZZAZIONE
# ============================================================
# Trasforma una frase in una lista di parole (token):
#  - tutto in minuscolo        -> "Il" e "il" diventano lo stesso token
#  - via la punteggiatura      -> "topo." diventa "topo"
#  - apostrofo come separatore -> "l'osso" diventa "l" + "osso"
def tokenizza(frase):
    return re.findall(r"[a-zàèéìíòóùú]+", frase.lower())

# ============================================================
# CONFIGURAZIONE
# ============================================================
EMBED_DIM      = 40
LEARNING_RATE  = 0.01
EPOCHS         = 90
WINDOW_SIZE    = 3
QUOTA_VERIFICA = 0.15   # 15% delle frasi NON usate nel training, servono per il test
SOGLIA_SUB     = 1e-3   # soglia del subsampling (valore di default di Word2Vec)

# ============================================================
# STEP 1: LEGGI VOCABOLARIO E FRASI DAI FILE
# ============================================================
with open("vocabolario.txt", "r", encoding="utf-8") as f:
    vocabolario = [riga.strip() for riga in f if riga.strip()]

with open("frasi.txt", "r", encoding="utf-8") as f:
    frasi_testo = [tokenizza(riga) for riga in f if riga.strip()]

parola2idx = {p: i for i, p in enumerate(vocabolario)}
idx2parola = {i: p for i, p in enumerate(vocabolario)}
VOCAB_SIZE  = len(vocabolario)

# Tieni solo le parole presenti nel vocabolario
frasi_pulite = []
token_totali, token_scartati = 0, 0
for frase in frasi_testo:
    parole = [p for p in frase if p in parola2idx]
    token_totali   += len(frase)
    token_scartati += len(frase) - len(parole)
    if len(parole) >= 2:
        frasi_pulite.append(parole)

print("=" * 60)
print("STEP 1: CARICAMENTO FILE")
print("=" * 60)
print(f"Vocabolario: {VOCAB_SIZE} parole")
print(f"Frasi:       {len(frasi_pulite)}")
print(f"Token letti: {token_totali} (scartati perché fuori vocabolario: {token_scartati})")

# ============================================================
# STEP 1b: DIVIDI LE FRASI IN TRAINING E VERIFICA
# ============================================================
# Il modello si allena solo sulle frasi di training.
# Le frasi di verifica non le vede mai: servono a capire se ha
# davvero imparato qualcosa o se ha solo memorizzato le frasi.
# Le frasi identiche finiscono sempre dalla stessa parte,
# altrimenti il test sarebbe su frasi già viste.
np.random.seed(42)

frasi_uniche = sorted({" ".join(f) for f in frasi_pulite})
np.random.shuffle(frasi_uniche)
n_verifica       = int(len(frasi_uniche) * QUOTA_VERIFICA)
insieme_verifica = set(frasi_uniche[:n_verifica])

frasi_training = [f for f in frasi_pulite if " ".join(f) not in insieme_verifica]
frasi_verifica = [f for f in frasi_pulite if " ".join(f) in insieme_verifica]

# Salva le due liste: il test le rilegge da qui
with open("frasi_training.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(" ".join(fr) for fr in frasi_training) + "\n")
with open("frasi_verifica.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(" ".join(fr) for fr in frasi_verifica) + "\n")

# Da parole a indici
frasi_training = [[parola2idx[p] for p in fr] for fr in frasi_training]
frasi_verifica = [[parola2idx[p] for p in fr] for fr in frasi_verifica]

print(f"Training:    {len(frasi_training)} frasi")
print(f"Verifica:    {len(frasi_verifica)} frasi (salvate in frasi_verifica.txt)")

# ============================================================
# STEP 2: PARAMETRI INIZIALI (casuali)
# ============================================================
embeddings = np.random.uniform(-0.5, 0.5, (VOCAB_SIZE, EMBED_DIM))
W_out      = np.random.uniform(-0.5, 0.5, (EMBED_DIM, VOCAB_SIZE))

print(f"\n{'=' * 60}")
print("STEP 2: PARAMETRI INIZIALI")
print("=" * 60)
print(f"Embedding matrix: {embeddings.shape}")
print(f"W_out matrix:     {W_out.shape}")

# ============================================================
# STEP 3: COPPIE (target, contesto)
# ============================================================
def genera_coppie(frasi, window_size):
    coppie = []
    for frase in frasi:
        for i, target in enumerate(frase):
            inizio   = max(0, i - window_size)
            fine     = min(len(frase), i + window_size + 1)
            contesto = [frase[j] for j in range(inizio, fine) if j != i]
            if contesto:
                coppie.append((target, contesto))
    return coppie

coppie          = genera_coppie(frasi_training, WINDOW_SIZE)
coppie_verifica = genera_coppie(frasi_verifica, WINDOW_SIZE)

print(f"\n{'=' * 60}")
print("STEP 3: COPPIE (target, contesto)")
print("=" * 60)
print(f"{len(coppie)} coppie di training, {len(coppie_verifica)} coppie di verifica.")

# ============================================================
# STEP 3b: SUBSAMPLING DELLE PAROLE FREQUENTI (opzionale)
# ============================================================
# Parole come "il", "la", "e" compaiono ovunque: il modello impara
# presto a indovinarle e da lì in poi insegnano poco, ma continuano
# a occupare gran parte del training.
# Word2Vec le scarta a caso: a ogni epoca ogni parola viene tenuta
# con una probabilità che scende quanto più la parola è frequente.
#
#   f = frequenza della parola (es. "il" = 0.11, cioè l'11% del testo)
#   probabilità di tenerla = sqrt(SOGLIA / f) + SOGLIA / f   (max 1)
#
# Le parole rare (f piccola) vengono tenute sempre.
# Le parole tolte spariscono dalla frase, quindi la finestra di contesto
# "si allarga" e arriva a parole più lontane e più informative.
conteggi = np.zeros(VOCAB_SIZE)
for frase in frasi_training:
    for idx in frase:
        conteggi[idx] += 1
frequenze = conteggi / conteggi.sum()

with np.errstate(divide="ignore"):
    prob_tenere = np.sqrt(SOGLIA_SUB / frequenze) + SOGLIA_SUB / frequenze
prob_tenere = np.minimum(prob_tenere, 1.0)

def sottocampiona(frasi):
    nuove = []
    for frase in frasi:
        tenute = [idx for idx in frase if np.random.random() < prob_tenere[idx]]
        if len(tenute) >= 2:
            nuove.append(tenute)
    return nuove

if SUBSAMPLING:
    print(f"\n{'=' * 60}")
    print("STEP 3b: SUBSAMPLING ATTIVO")
    print("=" * 60)
    print(f"  {'parola':<12}{'frequenza':>10}{'tenuta al':>12}")
    for idx in np.argsort(conteggi)[::-1][:10]:
        print(f"  {idx2parola[idx]:<12}{100 * frequenze[idx]:>9.1f}%{100 * prob_tenere[idx]:>11.0f}%")
    media = (prob_tenere * conteggi).sum() / conteggi.sum()
    print(f"  In media resta il {100 * media:.0f}% delle parole a ogni epoca.")

# ============================================================
# STEP 4: TRAINING
# ============================================================
def vettore_medio(contesto_idx):
    return embeddings[contesto_idx].mean(axis=0)

def calcola_punteggi(v_medio):
    return W_out.T @ v_medio

def softmax(scores):
    scores = scores - scores.max()
    exp    = np.exp(scores)
    return exp / exp.sum()

def cross_entropy_loss(probs, target_idx):
    return -np.log(probs[target_idx] + 1e-9)

def backward(v_medio, probs, target_idx, contesto_idx):
    global W_out, embeddings

    d_scores  = probs.copy()
    d_scores[target_idx] -= 1.0

    d_W_out   = np.outer(v_medio, d_scores)
    d_v_medio = W_out @ d_scores

    W_out -= LEARNING_RATE * d_W_out

    d_embed = d_v_medio / len(contesto_idx)
    for idx in contesto_idx:
        embeddings[idx] -= LEARNING_RATE * d_embed

def loss_verifica():
    # Solo previsione sulle frasi di verifica, nessun aggiornamento dei pesi
    totale = 0.0
    for target_idx, contesto_idx in coppie_verifica:
        probs   = softmax(calcola_punteggi(vettore_medio(contesto_idx)))
        totale += cross_entropy_loss(probs, target_idx)
    return totale / len(coppie_verifica)

print(f"\n{'=' * 60}")
print("STEP 4: TRAINING")
print("=" * 60)

if SALVA_LOSS:
    file_loss = open(FILE_LOSS, "w", encoding="utf-8")
    file_loss.write("epoca,loss_training,loss_verifica\n")

for epoch in range(1, EPOCHS + 1):
    if SUBSAMPLING:
        # a ogni epoca si scartano parole diverse
        coppie = genera_coppie(sottocampiona(frasi_training), WINDOW_SIZE)
    np.random.shuffle(coppie)
    loss_totale = 0.0

    for target_idx, contesto_idx in coppie:
        v_medio = vettore_medio(contesto_idx)
        scores  = calcola_punteggi(v_medio)
        probs   = softmax(scores)
        loss    = cross_entropy_loss(probs, target_idx)
        loss_totale += loss
        backward(v_medio, probs, target_idx, contesto_idx)

    loss_training = loss_totale / len(coppie)
    loss_ver      = loss_verifica()
    print(f"  Epoch {epoch} — loss training: {loss_training:.4f}   loss verifica: {loss_ver:.4f}")

    if SALVA_LOSS:
        file_loss.write(f"{epoch},{loss_training:.6f},{loss_ver:.6f}\n")
        file_loss.flush()   # scrive subito: se interrompi il training i dati restano

if SALVA_LOSS:
    file_loss.close()
    print(f"Loss salvata in {FILE_LOSS}")

np.save("embeddings.npy", embeddings)
np.save("W_out.npy", W_out)
print("Modello salvato.")
