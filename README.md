# SOLiGence Intelligent Coin Trading (IST) Platform

![image](https://github.com/user-attachments/assets/a04f5c7e-2a10-479a-9b6c-3481aeddc636)



## Website
The application is deployed and accessible at: [Soligence App · Streamlit](https://soligenceapp.streamlit.app/)

## Introduction

The SOLiGence Intelligent Coin Trading (IST) platform is an AI-powered cryptocurrency analysis system that combines machine learning with comprehensive market data to deliver actionable trading insights.

## ✨ Key Features

### Data Analysis
- Automated data acquisition from Yahoo Finance API
- 30+ major cryptocurrencies tracked
- 4 years of historical price/volume data
- Principal Component Analysis (PCA) and clustering

### Machine Learning
- **LSTM Neural Networks**: For temporal pattern recognition
- **XGBoost**: High-accuracy gradient boosted trees
- **Gradient Boosting**: Robust price trend prediction
- **Support Vector Regression**: Effective in volatile markets

### Visualization Tools
- Interactive candlestick charts
- Moving average analysis
- Correlation matrices
- Market state visualization
- Price distribution analysis

### Trading Features
- AI-generated buy/sell signals
- Profit-based coin selection
- Risk assessment metrics
- News aggregation from top sources

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DeborahAdedigba/Soligence_App.git
   cd Soligence_App/Web_page
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run Soli_app.py
   ```

## 🖥️ Usage

After launching the application:

1. Use the sidebar to navigate between sections:
   - **Home**: Platform overview
   - **Dataset**: Explore and filter cryptocurrency data
   - **Visualizations**: Interactive charts and technical analysis
   - **Predictions**: Model training and trading signals
   - **News**: Latest cryptocurrency market updates

2. Key workflows:
   - Generate moving average analyses
   - View correlation between different coins
   - Get AI-powered trading recommendations
   - Compare model performance metrics

## 📂 Project Structure

```
Soligence_App/
├── Web_page/
│   ├── Soli_app.py          # Main application file
│   ├── requirements.txt     # Python dependencies
│   ├── trained_models/      # Saved ML models
│   └── cached_models/       # Model cache
├── LICENSE
└── README.md
```

## 📊 Model Evaluation

Performance metrics tracked for all models:

| Metric | Description | Target |
|--------|-------------|--------|
| MAE | Mean Absolute Error | Minimize |
| RMSE | Root Mean Squared Error | Minimize |
| R² | R-squared coefficient | Maximize |
| MAPE | Mean Absolute Percentage Error | Minimize |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Deborah Adedigba**  
LinkedIn: [LinkedIn](https://www.linkedin.com/in/deborah-adedigba-bb917314b/)

