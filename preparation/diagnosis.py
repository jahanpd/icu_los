import os
import numpy as np
import pandas as pd
import voyageai
import pickle 
from pathlib import Path
from .autoencoder import Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim


vo = voyageai.Client(api_key=os.environ["VOYAGE_KEY"])

# get csv maps for code to free text
path1 = Path(__file__).parent / 'maps' / 'icu-apache-codes-ANZICS.csv'  
path2 = Path(__file__).parent / 'maps' / 'icu-apache-Subdiagnosis-codes-ANZICS.csv'

codes = pd.read_csv(path1)
subcodes = pd.read_csv(path2)

class EmbedDiagnosis():
    def __init__(
            self, 
            ):
        # iterate over the set of diagnostic
        # and subdiagnostic codes and create a dict
        # "diag-subdiag":"free text, free text"
        self.diagnoses_text = {}
        for _, row in subcodes.iterrows():
            filt = codes[codes.code == row.Code]
            codetext = filt[" text"].values[0]
            subcodetext = row.desc
            string = f"the diagnostic category is {codetext.strip()}, and the subgroup is {subcodetext.strip()}"
            self.diagnoses_text[row.subCode] = string
        for _, row in codes.iterrows():
            string = f"the diagnostic category is {row[' text'].strip()}"
            self.diagnoses_text[row.code] = string

        self.diagnoses_embed = None
        self.autoencoder = None
        self.missing = 0

        try:
            self.model = torch.load(f"{os.environ['OUT_DIR']}autoencoder.pth", weights_only=False)
        except Exception as e:
            print(e)
            print("No previous autoencoder model, please run train_embedding_model()")

        try:
            # get text embeddings
            self.diagnoses_embed = pickle.load(
                    open(f"{os.environ['OUT_DIR']}final_embeddings.pickle", 'rb'))
        except Exception as e:
            print(e)
            print("No previous small embeddings found, please run get_embeddings()")

        
    def get_voyage_embeddings(self):
        # try and read result from disk else
        try:
            result = pickle.load(open(f"{os.environ['OUT_DIR']}embed_result.pickle", "rb"))
            print("loaded big embeddings")
            return result
        except:
            texts = [value for _, value in self.diagnoses_text.items()]
            result = vo.embed(texts, model="voyage-3-large")
            big_embeddings = {}
            for subcode, embedding in zip(self.diagnoses_text.keys(), result.embeddings):
                big_embeddings[subcode] = embedding
            pickle.dump(big_embeddings, open(f"{os.environ['OUT_DIR']}embed_result.pickle", "wb"))
            print("called embeddings from voyage and processed")
            return big_embeddings


    def train_embedding_model(self):
        results = self.get_voyage_embeddings()

        data = torch.from_numpy(
                np.array([e for _, e in results.items()])).float()

        model = Autoencoder()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 6000
        for epoch in range(epochs):
            outputs = model(data)
            loss = criterion(outputs, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}')

        torch.save(model, f"{os.environ['OUT_DIR']}autoencoder.pth")

        self.model = model

    def get_embeddings(self):
        results = self.get_voyage_embeddings()
        final_embeddings = {}
        for subcode, big_embedding in results.items():
            sm_embedding = self.model.encoder(torch.tensor(big_embedding))
            final_embeddings[subcode] = sm_embedding.tolist()
        pickle.dump(final_embeddings, open(f"{os.environ['OUT_DIR']}final_embeddings.pickle", "wb"))
        self.diagnoses_embed = final_embeddings

    def return_small_embedding(self, code):
        assert self.diagnoses_embed is not None
        if code not in self.diagnoses_embed:
            try:
                return self.diagnoses_embed[float(int(code))]
            except Exception as e:
                self.missing += 1 # a very cheeky way of keeping track of this
                return [None]*8
        try:
            return self.diagnoses_embed[code]
        except Exception as e:
            self.missing += 1
            return [None]*8

# embed = EmbedDiagnosis()

# embed.train_embedding()

# embed.get_embeddings()

# print(embed.return_small_embedding(101.0))
