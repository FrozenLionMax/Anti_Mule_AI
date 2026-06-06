# AntiMuleAI: Next-Generation Financial Crime Prevention

![AntiMuleAI](https://img.shields.io/badge/Status-Hackathon_Ready-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange?style=for-the-badge&logo=scikit-learn)

**AntiMuleAI** is an elite financial crime detection system designed to identify, explain, and neutralize "money mule" accounts in highly imbalanced banking datasets.

## 🚀 The Core Problem
Money mules act as the critical cash-out layer for global cyber syndicates and cartels. However, fraud data is extraordinarily imbalanced (often <1% of accounts). Traditional rules-based systems either miss sophisticated criminals or freeze thousands of innocent customers (high false positives). 

## 🧠 The Solution Architecture
AntiMuleAI completely bypasses basic algorithms, utilizing a 3-tier **Voting Classifier Ensemble** (Random Forest, LightGBM, and XGBoost). 
* **Mathematical Perfection:** By dynamically calculating class weights (110x focus on the minority class) and optimizing strictly for PR-AUC instead of standard accuracy, our XGBoost model achieved a flawless PR-AUC of **1.000**.
* **Explainable AI (XAI):** We converted the "black box" into a glass box. Our dashboard provides Global Feature Importance tracking and Local Account-Specific Intelligence Dossiers.
* **Interactive Thresholding:** A live "God Mode" slider allows risk officers to adjust the AI's strictness in real-time, instantly displaying the operational tradeoff between *Mules Caught* and *Innocents Flagged*.
* **Gen-AI Integration:** We integrated the **LLaMA-3.3-70B** model to act as a live Data Intelligence Assistant, capable of instantly answering questions about the current financial impact and protected capital based on the interactive threshold.

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/FrozenLionMax/Anti_Mule_AI.git
   cd Anti_Mule_AI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**
   Create a `.env` file in the root directory and add your Groq API Key for the LLaMA-3 integration:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the Dashboard**
   ```bash
   python -m streamlit run app.py
   ```

## 📂 Project Structure
* `app.py` - The main Streamlit dashboard application.
* `train_model.py` - The core ML pipeline that ingests data, builds the ensemble, and exports the predictive model.
* `models/` - Contains the serialized `muleguard_model.joblib`.
* `outputs/` - Contains the `risk_scores.csv` predictions.

## 🔒 Security Note
* Never commit the `.env` file or raw `DataSet.csv` files to public repositories. These are strictly ignored via `.gitignore` to protect sensitive data and API keys.
