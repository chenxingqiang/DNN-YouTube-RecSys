# YouTube-DNN-RecSys: Deep Neural Networks for YouTube Recommendations

A clean, well-organized implementation of Deep Neural Networks for YouTube Recommendations, featuring both Python (TensorFlow) and Scala (Spark) implementations.

## 🏗️ Project Structure

```
DNN-YouTube-RecSys/
├── src/
│   ├── python/           # Python implementation using TensorFlow
│   │   ├── models/       # Core DNN model and loading utilities
│   │   ├── data/         # Data processing and TFRecords handling
│   │   ├── utils/        # TensorBoard and utility functions
│   │   ├── examples/     # Usage examples and tutorials
│   │   └── reference/    # Custom layers and feature engineering
│   └── scala/            # Scala implementation using Spark
│       ├── models/       # Feature building and embedding models
│       ├── data/         # Data generation scripts
│       ├── prediction/   # User vector and item embedding prediction
│       ├── core/         # Base Spark application classes
│       ├── examples/     # Spark usage examples
│       └── config/       # Environment configuration files
├── data/                 # Data storage and model artifacts
│   ├── tfrecords/        # Training and evaluation data
│   └── checkpoints/      # Model checkpoints and saved models
├── tests/                # Test suites for both Python and Scala
├── docs/                 # Research paper and documentation
├── requirements.txt      # Python dependencies
├── pom.xml              # Maven configuration for Scala
└── .gitignore           # Git ignore patterns
```

## 🚀 Quick Start

### Python (TensorFlow) Implementation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   cd src/python
   python models/dnn.py
   ```

3. **Run examples:**
   ```bash
   python examples/example1.py
   python examples/example2.py
   ```

### Scala (Spark) Implementation

1. **Build the project:**
   ```bash
   mvn clean compile
   ```

2. **Run examples:**
   ```bash
   mvn exec:java -Dexec.mainClass="example.Example1"
   ```

## 📚 Key Components

### Python Implementation
- **`models/dnn.py`**: Core deep neural network model
- **`models/load_dnn_model.py`**: Model loading and inference utilities
- **`data/data2tfrecords.py`**: Data conversion to TFRecords format
- **`utils/tensor_board.py`**: TensorBoard integration for training visualization

### Scala Implementation
- **`models/FeatureBuilder.scala`**: Feature engineering utilities
- **`prediction/PredictUserVector.scala`**: User vector prediction
- **`prediction/ItemEmbeddingPredictor.scala`**: Item embedding generation
- **`core/BaseSparkLocal.scala`**: Local Spark application base class

## 🔧 Configuration

- **Python**: Configure via `requirements.txt` and environment variables
- **Scala**: Configure via `src/scala/config/` properties files
- **Data**: Store training data in `data/tfrecords/` directory
- **Models**: Save checkpoints in `data/checkpoints/` directory

## 📖 Documentation

- **Research Paper**: `docs/Deep Neural Networks for YouTube Recommendations.pdf`
- **Code Examples**: See `src/python/examples/` and `src/scala/examples/`
- **Reference Implementations**: Check `src/python/reference/` for custom components

## 🤝 Contributing

1. Follow the established directory structure
2. Add tests in the appropriate `tests/` subdirectory
3. Update documentation for any new features
4. Ensure both Python and Scala implementations remain consistent

## 📄 License

This project implements the research described in "Deep Neural Networks for YouTube Recommendations" paper. Please refer to the original paper for academic citations and research context.
