# AI-Optimized Resource Allocation in Smart City Networks: A Comparative Study of Schedulers in 4G and 5G Networks Using NS-3 Simulation and Machine Learning

## Project Overview

Smart city networks support diverse applications such as the Internet of Things (IoT), video streaming, and other data-intensive services. These applications have different Quality of Service (QoS) requirements, making efficient radio resource allocation essential.

This project compares conventional scheduling algorithms with an AI-based resource allocation approach in 4G LTE and 5G NR networks using NS-3 simulations and machine learning.

The proposed approach uses an XGBoost regression model trained on simulation-generated network data to predict QoS-related labels and support traffic-aware resource allocation.

## Objectives

The primary objective of this project is to simulate 4G LTE and 5G NR networks using NS-3 and evaluate traditional scheduling algorithms under different network loads and traffic conditions. The project also aims to develop an AI-based scheduler using simulation-generated data, compare its performance with conventional scheduling approaches, and investigate QoS-aware resource allocation for voice, video, and IoT traffic.

## Technologies Used

The project uses NS-3 for LTE and 5G NR network simulation and Python for data preprocessing, analysis, and model development. XGBoost is used for machine learning regression, while NumPy and Pandas support data processing and manipulation. Scikit-learn is used for model evaluation and regression metrics, and Matplotlib and Seaborn are used for data visualization. The ns-3-nr module supports 5G NR network simulation.

## Methodology

The methodology begins with configuring LTE and 5G NR network environments with user equipment, mobility, and channel conditions. IoT, video, and other traffic types are simulated with different QoS requirements.

Conventional scheduling algorithms, including Round Robin (RR), Proportional Fair (PF), and Maximum Throughput (MT), are evaluated alongside the QoS scheduler used in the 5G experiments. Network measurements such as throughput, delay, queue size, and Signal-to-Interference-plus-Noise Ratio (SINR) are collected from simulation outputs and traces.

The generated data is processed using Python and used to train an XGBoost regression model that learns relationships between network parameters and QoS labels. The model is evaluated using regression metrics, and its performance is compared with conventional scheduling approaches across LTE and 5G NR environments.

## Machine Learning Model

The proposed AI-based scheduler uses XGBoost Regression with 100 estimators, a maximum tree depth of 6, and a learning rate of 0.1. Subsampling is used to support generalization, and the objective is to minimize squared error.

Model performance is evaluated using the R² score, Root Mean Square Error (RMSE), and Mean Absolute Error (MAE). These metrics help assess the model's regression fit and prediction accuracy.

## Results and Performance

### LTE Scheduler Evaluation

The LTE simulation results show that Maximum Throughput (MT) achieved approximately 50 Mbps at 100 user equipment (UEs). Proportional Fair (PF) achieved approximately 45 Mbps, while Round Robin (RR) achieved approximately 44 Mbps at the same user density.

Although MT achieved higher throughput, it experienced increased delay, packet loss, and reduced fairness as network load increased. PF and RR demonstrated comparatively better fairness and delay performance in the reported scenarios.

### LTE AI Model

The LTE AI regression model was trained using a dataset containing 83,820 samples. The model achieved an RMSE of approximately 0.021 and an MAE of approximately 0.016. The reported R² values for voice traffic were 0.9431, 0.9441, and 0.9757 across the evaluated model comparisons.

The report also describes substantial reductions in prediction error compared with conventional RR scheduling.

### 5G NR AI Model

The 5G NR AI model demonstrated high regression performance across the evaluated traffic classes. The reported R² scores were 0.993 for voice traffic, 0.998 for video traffic, and 0.997 for IoT traffic.

The reported RMSE ranged from approximately 0.027 to 0.067, while the MAE ranged from approximately 0.020 to 0.025.

The resource allocation analysis showed that the AI-based approach assigned more resources to voice and video traffic than conventional schedulers in the evaluated scenarios.

These values represent the results reported in the project report and describe the evaluated simulation and regression experiments, rather than measurements from a live deployed network.

## Key Performance Indicators

The project evaluates network performance using throughput, latency, delay, jitter, Packet Loss Ratio (PLR), Jain's Fairness Index, and Signal-to-Interference-plus-Noise Ratio (SINR). Regression performance is evaluated using the R² score, Root Mean Square Error (RMSE), and Mean Absolute Error (MAE).

## Applications

The proposed approach is relevant to smart city communication networks, QoS-aware wireless resource management, IoT connectivity, video streaming services, traffic-aware LTE and 5G scheduling, and machine learning-assisted radio resource allocation.

## Conclusion

This project investigates how machine learning can improve resource allocation in heterogeneous smart city networks. By combining NS-3 network simulations with XGBoost regression, the study compares conventional scheduling approaches with an AI-based, QoS-aware alternative across 4G LTE and 5G NR environments.

The reported results demonstrate strong regression performance and traffic-aware allocation behavior, highlighting the potential of AI-assisted scheduling for future wireless networks.

## Project Team

**Ashwika S, Lakshmi Sanjeev, and Kirthana S**

The project was carried out at the School of Electronics Engineering (SENSE), Vellore Institute of Technology (VIT), Chennai, as part of the Bachelor of Technology degree in Electronics and Communication Engineering.

**Project Guide:** Dr. Hemanth C

**Project Report:** April 2026

## References

The project draws on published research covering fairness in cognitive radio networks, QoS-aware scheduling, 5G network slicing, machine learning-based resource allocation, and emerging wireless communication technologies. The complete bibliography is available in the project's report.
