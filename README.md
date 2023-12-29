
#  xw74ByaPqC9WCASm

A machine learning powered pipeline that could spot talented individuals, and rank them based on their fitness.

## Installation

```bash
cd p3
poetry install
```

## Usage
La pipeline contient trois modules:
1. Permet de calculer semi manuellement les scores de fitness en utilisant le csv fourni
2. Permet d'entrainer un premier modele avec les score calculés, et
3. Permet de suivre le fichier des scores, et de déclencher à chaque modification (starring) le retrainement du modele. L'aopprentissage est incremental, donc on reutiliser le plus recent model pour commencer.

## 1. Fitness scores
Le fitness score est une moyenne pondere de e. La ponderation par d.faut utiliser est .9 et .1. Personnellement je pense que le job title (experience de travail) est plus importante que le nombre de connection qui est pour moi un plus. Entre deux candidats, je prefere le candidat avec le plus d'experience même s'il a moins de connection. Mais les poids ne sont que des parametres, et peuvent etre ajustés.
fitness =  0.9 x importance_job_tile + 0.1 x importance_connection.
importance_job_tile =  cosine (job_title[vec], query[vec]) (betwen 0..1)
importance_connection =  sklearn.MINMAX(connection_COL, min=0.5, max=1) (between 0.5..1)

COL[vec] =  EMBEDDER(preprocessing(COL)). Embedder est fixe avec l'option -en/-et pour choisir un modele pretrainé de hugging face ou un vectorizer de sklearn comme countVctorizer ou Tdif

```bash
(p3-py3.11) (base) \p3>poetry run cli -f  ".data\potential-talents - Aspiring human resources - seeking human resources.csv" -en albert-base-v2 -q "Aspiring human resources" --debug rank
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\..\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package wordnet to
[nltk_data]     C:\Users\..\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\..\nltk_data...
[nltk_data]   Package punkt is already up-to-date!

| id | job_title                          | location    | connection | fit      | query                     | qId |
|----|------------------------------------|-------------|------------|----------|---------------------------|-----|
| 8  | HR Senior Specialist               | SF Bay Area | 500+       | 0.919851 | Aspiring human resources  | 0   |
| 6  | Aspiring HR Specialist             | NYC Area    | 1          | 0.918608 | Aspiring human resources  | 0   |
| 10 | Seeking HRIS and Generalist        | Philadelphia Area | 500+ | 0.891796 | Aspiring human resources  | 0   |
| 4  | People Development Coordinator     | Denton, Texas| 500+      | 0.889278 | Aspiring human resources  | 0   |
| 2  | Native English Teacher             | Kanada      | 500+       | 0.889178 | Aspiring human resources  | 0   |
| 5  | Advisory Board Member              | İzmir, Türkiye| 500+    | 0.884010 | Aspiring human resources  | 0   |
| 7  | Student at Humber College          | Kanada     | 61         | 0.873881 | Aspiring human resources  | 0   |
| 9  | Student at Humber College          | Kanada     | 61         | 0.873881 | Aspiring human resources  | 0   |
| 3  | Aspiring HR Professiona| Raleigh-Durham, NC Area | 44         | 0.871002 | Aspiring human resources  | 0   |
| 1  | 2019 Bauer College Grad         | Houston, Texas | 85         | 0.844078 | Aspiring human resources  | 0   |

```

## 2. XGBoost ranker
Le modele utiliser vient de xgboost, c'est xgboost Ranker avec des parametres par défaut.
```bash
(p3-py3.11) (base) \p3>poetry run cli -f  "test.csv" -en albert-base-v2  --debug fit
| id | job_title                          | location                  | conn | fit| query                     | qId | relevance_score  |
|----|------------------------------------|---------------------------|------------|----|---------------------------|------------------------|
| 8  | HR Senior Specialist               | San Francisco Bay Area    | 500+ | 0.919851 | Aspiring human resources  | 0   | 1                |
| 6  | Aspiring Human Resources Specialist| Greater New York City Area| 1    | 0.918608 | Aspiring human resources  | 0   | 2                |
| 10 | Seeking Human Resources HRIS ...   | Greater Philadelphia Area | 500+ | 0.891796 | Aspiring human resources  | 0   | 3                |
| 4  | People Development Coordinator ... | Denton, Texas             | 500+ | 0.889278 | Aspiring human resources  | 0   | 4                |
| 2  | Native English Teacher at EPIK ... | Kanada                    | 500+ | 0.889178 | Aspiring human resources  | 0   | 5                |


Most recent model saved at : ..\.models\model_20231229064819.json
Model fitted with :
        objective : rank:ndcg
        random_state : 42
        tree_method : hist

Result: rank:ndcg -> 0.5443434449057798

Model saved at : ..\.models/model_20231229065022.json
```
## 3. WatchDog
Periodically retrain the model with new data et enregistrer la sortie dans un fichier journal.

## RECAP
1. Success Metric(s):

-[X] Rank candidates based on a fitness score.
-[X] Re-rank candidates when a candidate is starred.

2. Bonus(es):

- [X] We are interested in a robust algorithm, tell us how your solution works and show us how your ranking gets better with each starring action.
  Implemented : Watchdog to know when to retrain the model. Starring = modifie le fichier csv en mettant un fit score de 1 à la personne choisi et en le déplacant en haut de la liste.

- [] How can we filter out candidates which in the first place should not be in this list? Can we determine a cut-off point that would work for other roles without losing high potential candidates?

-[x] Do you have any ideas that we should explore so that we can even automate this procedure to 
  Answer : The process is already automate. The model is re-trained based on the starring process

## Next steps
- [] Finir la documentation des fonctions
- [] Hyperparameter Tuning option 
- [] Train-test split instead of training on the full csv
- [] When watching the file, give more flexibility for the embedder