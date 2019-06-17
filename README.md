# Text-to-SQL Development Environment

### Models
- SyntaxSQL
- TypeSQL
- SQLNet
- From Predictor

### Dataset
- Spider 

### Settings
- Download glove from [Glove](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) and put it under glove/
- Download modified Spider dataset from [Spider](https://drive.google.com/file/d/1TsekxtgIUum4xa6WRGFUGS_jpPWhvamL/view?usp=sharing) and put it under datasets/spider/data/
- Download trained model weights from ----- and put it under saved_models/{model_name}/


#### Fixed bugs for the Spider Dataset
- Order difference b/w ("column names", "column names original") and ("table names", "table names original") for db_id: "scholar", "store_1", and "formula_1"
- Non-existing "Ref_Company_Types" table being used for db_id: assets_maintenance
- "Nested query in from clause" bug. (About 7 queries are erased)
- "syntaxsqlnet bug - parsing bug" bug. (About 50 queries are erased)
