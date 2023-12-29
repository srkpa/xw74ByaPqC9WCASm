
#  xw74ByaPqC9WCASm

A machine learning powered pipeline that could spot talented individuals, and rank them based on their fitness.

## Installation

```bash
cd p3
poetry install
```

## Usage
The pipeline comprises three key modules:
1. Manually calculate fitness scores using the provided CSV.
2. Train an initial model with the calculated fitness scores.
3. Monitor the csv file and trigger retraining of the model upon modification (starring). Learning is incremental, so the most recent model is reused to start the training.

## 1. Fitness scores
The fitness score is computed as a weighted average with default weights of 0.9 and 0.1, but these weights can be adjusted. In my personal perspective, I assign higher importance to job title (work experience) compared to the number of connections. Given two candidates, I favor the one with more experience, even if they have fewer connections.

The fitness score formula is given by:

> fitness = 0.9 × importance_job_tile + 0.1 × importance_connection

The importance of job title is determined using cosine similarity between the job title vector and the query vector, ranging from 0 to 1:

> importance_job_tile =  cosine (job_title[vec], query[vec])

Connection importance is scaled using Min-Max scaling on the connection column (ranging from 0.5 to 1).
> importance_connection =  sklearn.MINMAX(connection_COL, min=0.5, max=1) 
with :
> COL[vec] = EMBEDDER(preprocessing(COL)). 

The Embedder is choosed with the option ```−en/−et``` to choose a pretrained model from Hugging Face or a vectorizer from sklearn, such as CountVectorizer or TfidfVectorizer.

The command to run this module is as follows:

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

We utilize an XGBoost Ranker with default parameters and train the model with the fitness scores.

The command to run this module is as follows:
```bash
(p3-py3.11) (base) \p3>poetry run cli -f  "test.csv" -en albert-base-v2  --debug fit
| id | job_title                          | location                  | conn | fit| query                     | qId | relevance_score  |
|----|------------------------------------|---------------------------|------------|----|---------------------------|------------------|
| 8  | HR Senior Specialist               | San Francisco Bay Area    | 500+ | 0.919851 | Aspiring human resources  | 0   | 1          |
| 6  | Aspiring Human Resources Specialist| Greater New York City Area| 1    | 0.918608 | Aspiring human resources  | 0   | 2          |
| 10 | Seeking Human Resources HRIS ...   | Greater Philadelphia Area | 500+ | 0.891796 | Aspiring human resources  | 0   | 3          |
| 4  | People Development Coordinator ... | Denton, Texas             | 500+ | 0.889278 | Aspiring human resources  | 0   | 4          |
| 2  | Native English Teacher at EPIK ... | Kanada                    | 500+ | 0.889178 | Aspiring human resources  | 0   | 5          |


Most recent model saved at : ..\.models\model_20231229064819.json
Model fitted with :
        objective : rank:ndcg
        random_state : 42
        tree_method : hist

Result: rank:ndcg -> 0.5443434449057798

Model saved at : ..\.models/model_20231229065022.json
```
## 3. WatchDog

Periodically retrain the model with new data and save the output to a log file.

```bash
poetry run watch
```
The log file store all the timestamps where the file has been modified and the model retrain
```
2023-12-29 08:36:40
2023-12-29 08:37:13
```

## SUMMARY
1. Success Metric(s):

- [x] Rank candidates based on a fitness score.
- [x] Re-rank candidates when a candidate is starred.

2. Bonus(es):

- [x] We are interested in a robust algorithm, tell us how your solution works...
  Answer : Done
  
- [x]: ...and show us how your ranking gets better with each starring action. 
  Answer : Watchdog is employed to monitor changes in the dataset, indicating when it is time to retrain the model. The 'Starring' process here involves modifying the CSV file, assigning a fit score of 1 to the selected individual, and relocating them to the top of the list. We might decide not to save the most recent model (as done for now) if its score is not better than the previous one, for example."

- [] How can we filter out candidates which in the first place should not be in this list? Can we determine a cut-off point that would work for other roles without losing high potential candidates?
  Answer: 

- [x] An automated process; the model is retrained based on starring actions. 
  Answer : The process is already automated. The model is re-trained based on the starring process.

## Next steps
 - [] Complete the documentation for functions.
 - [] Implement hyperparameter tuning options.
 - [] Introduce train-test split instead of training on the full CSV.
 - [] Enhance flexibility for the embedder when watching the file.
 - [] Make watchdog (module 3) run in the background