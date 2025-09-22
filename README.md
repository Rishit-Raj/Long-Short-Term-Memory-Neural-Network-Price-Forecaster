# LSTM Price Forecaster — README

This README gives a **simple explanation of what an LSTM is** and **easy steps** to run the program.

---

## 1. What is an LSTM?

LSTM = **Long Short-Term Memory**. It is a type of neural network that works well with **time series data** (like stock prices).

* Normal neural networks treat inputs as independent. But in time series, the **past matters**.
* LSTMs have a **memory cell** that can remember useful patterns for a long time.
* They use **gates** (small switches) to decide:

  * what to keep,
  * what to forget,
  * what to output.

This makes LSTMs very good at spotting **trends and patterns over time**.

---

## 2. Why use LSTM for prices?

* Stock prices and indicators are **sequences**.
* LSTMs can learn from these sequences.
* They can model **momentum, cycles, and dependencies** better than basic models.

---

## 3. Project Structure

```
README.md          # this file
requirements.txt   # Python libraries needed
init.py            # initialization
dataset.py         # dataset
helpers.py         # helper function
model.py           # model
preprocessors.py   # pre-run requirements
evaluate.py        # test the model
main.py            # main run file
```

---

## 4. Setup

1. Install Python 3.8+.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.\.venv\Scripts\Activate.ps1 # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
pandas
scikit-learn
tensorflow
matplotlib
```

---

## 5. Data

Put your CSV file inside `data/`.

Example format:

```
date,feature1,feature2,...,target
2023-01-01,100,200,...,101
2023-01-02,101,201,...,103
```

* `date`: time step
* `features`: input variables
* `target`: value you want to predict

---

## 6. Train the Model

Run:

```bash
python src/train.py --data data/prices.csv --seq_len 50 --epochs 50 --batch_size 32 --model_dir checkpoints/
```

This will:

* Load the CSV
* Create sequences of length 50
* Train an LSTM
* Save the model in `checkpoints/`

---

## 7. Evaluate the Model

```bash
python src/evaluate.py --model checkpoints/best_model.h5 --data data/prices.csv
```

This checks performance on test data.

---

## 8. Make Predictions

```bash
python src/predict.py --model checkpoints/best_model.h5 --input data/recent.csv --output preds.csv
```

This creates predictions in a new CSV file.

---

## 9. Tips

* Change `seq_len` (like 20, 50, 100) to see what works best.
* Use more `epochs` for better accuracy (but longer training).
* Always test on unseen data.

---

## 10. License

Free to use and share.
