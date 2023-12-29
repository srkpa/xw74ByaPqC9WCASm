#
Potential Talents

#  xw74ByaPqC9WCASm

a machine learning powered pipeline that could spot talented individuals, and rank them based on their fitness
## Installation

Describe how to install your project.

```bash
cd p3
poetry install
```

# Usage
```bash
(p3-py3.11) (base) \p3>poetry run cli -f  ".data\potential-talents - Aspiring human resources - seeking human resources.csv" -en albert-base-v2 -q "Aspiring human resources" --debug rank
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\kpano\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package wordnet to
[nltk_data]     C:\Users\kpano\AppData\Roaming\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\kpano\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!

| id | job_title                          | location    | connection | fit      | query                     | qId |
|----|------------------------------------|-------------|------------|----------|---------------------------|-----|
| 8  | HR Senior Specialist               | SF Bay Area | 500+       | 0.919851 | Aspiring human resources  | 0   |
| 6  | Aspiring HR Specialist             | NYC Area    | 1          | 0.918608 | Aspiring human resources  | 0   |
| 10 | Seeking HRIS and Generalist        | Philadelphia Area | 500+ | 0.891796 | Aspiring human resources  | 0   |
| 4  | People Development Coordinator    | Denton, Texas| 500+       | 0.889278 | Aspiring human resources  | 0   |
| 2  | Native English Teacher             | Kanada      | 500+       | 0.889178 | Aspiring human resources  | 0   |
| 5  | Advisory Board Member               | İzmir, Türkiye| 500+    | 0.884010 | Aspiring human resources  | 0   |
| 7  | Student at Humber College           | Kanada     | 61         | 0.873881 | Aspiring human resources  | 0   |
| 9  | Student at Humber College           | Kanada     | 61         | 0.873881 | Aspiring human resources  | 0   |
| 3  | Aspiring HR Professiona| Raleigh-Durham, NC Area | 44         | 0.871002 | Aspiring human resources  | 0   |
| 1  | 2019 Bauer College Grad         | Houston, Texas | 85         | 0.844078 | Aspiring human resources  | 0   |

```


