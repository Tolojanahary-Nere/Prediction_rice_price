from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import datetime
import pandas as pd
import random
from datetime import date, timedelta

# Connexion MongoDB
client = MongoClient("mongodb://mongo:27017/")
db = client["rice_db"]
collection = db["prices"]

# App FastAPI
app = FastAPI(title="API Prix du Riz Madagascar")

# Configuration CORS
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schéma Pydantic
class RicePrice(BaseModel):
    date: datetime.date
    region: str
    type: str
    price: float

@app.post("/prices/")
def add_price(price: RicePrice):
    doc = price.dict()
    doc["date"] = doc["date"].isoformat()
    result = collection.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return {"msg": "Prix ajouté avec succès", "data": doc}

@app.get("/prices/")
def get_prices(limit: int = 100):  # Ajout pagination simple
    data = []
    for item in collection.find({}).limit(limit).sort("date", -1):
        item["_id"] = str(item["_id"])
        data.append(item)
    return {"data": data}

@app.get("/stats/")
def get_stats():
    pipeline = [
        {
            "$group": {
                "_id": {"region": "$region", "type": "$type", "month": {"$dateToString": {"format": "%Y-%m", "date": {"$toDate": "$date"}}}},
                "avg_price": {"$avg": "$price"},
                "min_price": {"$min": "$price"},
                "max_price": {"$max": "$price"}
            }
        },
        {"$sort": {"_id.month": -1}}
    ]
    results = list(collection.aggregate(pipeline))

    # Transformer en format plus lisible
    stats = [
        {
            "region": r["_id"]["region"],
            "type": r["_id"]["type"],
            "month": r["_id"]["month"],
            "avg_price": round(r["avg_price"], 2),
            "min_price": round(r["min_price"], 2),
            "max_price": round(r["max_price"], 2)
        }
        for r in results
    ]
    return {"stats": stats}

# Endpoint pour upload dynamique CSV (nouveau !)
@app.post("/upload-csv/")
def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Seul les fichiers CSV sont acceptés")
    
    try:
        # Lecture en mémoire avec Pandas
        df = pd.read_csv(file.file)
        
        # Nettoyage basique (adapte aux colonnes de ton CSV)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['date', 'region', 'type', 'price'])  # Assume ces colonnes existent
        
        # Conversion en docs MongoDB
        docs = df.to_dict('records')
        
        # Insertion en batch
        if docs:
            # Nettoie les dates pour Mongo (string ISO)
            for doc in docs:
                doc["date"] = doc["date"].isoformat()
            result = collection.insert_many(docs)
            return {"msg": f"Upload réussi ! {len(result.inserted_ids)} lignes ajoutées."}
        else:
            raise HTTPException(status_code=400, detail="CSV vide ou invalide après nettoyage")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'upload : {str(e)}")

# Endpoint pour générer des seed data aléatoires (bonus)
@app.post("/generate-seed/{n}")
def generate_seed(n: int):
    seeds = []
    regions = ["Antananarivo", "Toamasina", "Fianarantsoa"]  # Exemples
    types = ["blanc", "rouge", "importé"]
    start_date = date(2023, 1, 1)
    
    for _ in range(n):
        doc = {
            "date": (start_date + timedelta(days=random.randint(0, 365))).isoformat(),
            "region": random.choice(regions),
            "type": random.choice(types),
            "price": round(random.uniform(2000, 5000), 2)
        }
        seeds.append(doc)
    
    result = collection.insert_many(seeds)
    return {"msg": f"{n} données seed générées."}

# Bonne pratique : Fermeture client à la shutdown
@app.on_event("shutdown")
def shutdown():
    client.close()