import spacy
from spacy.training import Example

# **GiNZAベースのカスタムモデルを作成**
nlp = spacy.load("ja_ginza")
ner = nlp.get_pipe("ner")

# **カスタムNER学習用データセット**
TRAIN_DATA = [
    ("橋本病の研究が進んでいる", {"entities": [(0, 3, "MISC")]}),
    ("川崎病の新しい治療法", {"entities": [(0, 3, "MISC")]}),
    ("小柳・原田病は自己免疫疾患である", {"entities": [(0, 7, "MISC")]}),
    ("菊池病はリンパ節に関連する", {"entities": [(0, 3, "MISC")]}),
    ("鈴木教授が高安動脈炎の研究を行った", {"entities": [(0, 4, "Person")]}),
    ("山田先生が糖尿病の臨床試験を担当した", {"entities": [(0, 4, "Person")]}),
]

# **カスタムNERデータの追加**
for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])  # MISC, Person を追加

# **トレーニングデータの作成**
examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in TRAIN_DATA]

# **学習**
optimizer = nlp.resume_training()
for i in range(10):  # エポック数
    for example in examples:
        nlp.update([example], drop=0.5, losses={})

# **学習済みモデルの保存**
nlp.to_disk("ja_ginza_custom")

print("カスタム学習済みモデル `ja_ginza_custom` を保存しました！")
