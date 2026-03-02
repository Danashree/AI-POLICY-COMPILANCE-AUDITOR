import chromadb
from sentence_transformers import SentenceTransformer
import pypdf
import os


pdf_path = os.path.join(os.getcwd(), "COMPANY HR POLICY HANDBOOK.pdf")

reader = pypdf.PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + " "

text = text.replace("\n", " ").strip()


CHUNK_SIZE = 1500

chunks = [
    text[i:i+CHUNK_SIZE]
    for i in range(0, len(text), CHUNK_SIZE)
]

print("Total chunks:", len(chunks))



model = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.Client()

collection = client.get_or_create_collection(
    name="policy_db",
    metadata={"hnsw:space": "cosine"}
)

# Clear old data
try:
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
except:
    pass


for i, chunk in enumerate(chunks):

    embedding = model.encode(chunk).tolist()

    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[str(i)]
    )

print("R1 Completed: Stored in ChromaDB")


TOP_K = 3
THRESHOLD = 0.6



def extract_relevant_sentence(text, query):

    clean = text.replace("\n", ".").replace("?", ".").replace("!", ".")
    sentences = clean.split(".")

    query_words = query.lower().split()

    best_sentence = ""
    best_score = -1

    for sentence in sentences:

        s = sentence.lower().strip()

        if s == "":
            continue

        score = 0

        # Strong priority rules
        if "employees receive" in s:
            score += 100

        if "earned leave" in s:
            score += 80

        if "per year" in s:
            score += 60

        if "credited monthly" in s:
            score += 40

        # Query match score
        for word in query_words:
            if word in s:
                score += 10

        # Number bonus
        if any(char.isdigit() for char in s):
            score += 20

        if score > best_score:
            best_score = score
            best_sentence = sentence.strip()

    # FINAL CLEANING: remove prefixes
    if "employees receive" in best_sentence.lower():
        index = best_sentence.lower().find("employees receive")
        best_sentence = best_sentence[index:]

    return best_sentence



def check_policy_question(query):

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    top_distance = distances[0]
    confidence = max(0, 1 - top_distance)

    print("\n==============================")
    print("Question:", query)
    print("Distance:", round(top_distance, 4))
    print("Confidence:", round(confidence, 4))

    if top_distance <= THRESHOLD:

        print("\nStatus: ANSWERED")

        best_answer = ""
        best_score = -1
        best_chunk = ""

        # Search all Top-K chunks
        for chunk in documents:

            sentence = extract_relevant_sentence(chunk, query)
            s = sentence.lower()

            score = 0

            if "employees receive" in s:
                score += 100

            if "earned leave" in s:
                score += 80

            if "per year" in s:
                score += 60

            if any(char.isdigit() for char in s):
                score += 20

            for word in query.lower().split():
                if word in s:
                    score += 10

            if score > best_score:
                best_score = score
                best_answer = sentence
                best_chunk = chunk

        print("\nFinal Answer:")
        print(best_answer)

        print("\nCitation:")
        print(best_chunk[:200], "...")

    else:

        print("\nStatus: NOT ANSWERED")
        print("Smart Response:")
        print("This question is not clearly covered in the HR policy document.")

    print("\nTop Results:")

    for i in range(len(documents)):

        conf = max(0, 1 - distances[i])

        print("\nResult", i+1)
        print("Confidence:", round(conf, 4))
        print(documents[i][:150], "...")




question = input("\nEnter Employee Question: ")


check_policy_question(question)
