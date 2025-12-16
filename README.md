# Customer Shopping Data Mining Project

**Đề tài Khai phá Dữ liệu - Data Mining Course**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Cấu trúc Project](#cấu-trúc-project)
- [Nhóm thực hiện](#nhóm-thực-hiện)

---

##  Giới thiệu

Dự án này thực hiện phân tích toàn diện dữ liệu mua sắm từ 10 trung tâm thương mại tại Istanbul, áp dụng các kỹ thuật khai phá dữ liệu bao gồm:

- **Clustering**: Phân nhóm khách hàng dựa trên RFM (Recency, Frequency, Monetary)
- **Association Rules**: Tìm kiếm mối quan hệ giữa các sản phẩm (Market Basket Analysis)
- **Classification**: Dự đoán danh mục sản phẩm dựa trên đặc điểm khách hàng

###  Mục tiêu

1. Khám phá patterns ẩn trong hành vi mua sắm của khách hàng
2. Phân nhóm khách hàng để cá nhân hóa marketing
3. Tìm ra product combinations cho cross-selling
4. Dự đoán xu hướng mua sắm dựa trên demographics

---

##  Dataset

**Nguồn**: [Customer Shopping Dataset - Kaggle](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset/data)

**Mô tả**: 
- Dữ liệu giao dịch từ 10 shopping malls ở Istanbul
- Thời gian: 2021-2023
- Số giao dịch: ~99,457 transactions

**Thuộc tính**:
| Column | Mô tả | Type |
|--------|-------|------|
| invoice_no | Mã hóa đơn (I + 6 digits) | String |
| customer_id | Mã khách hàng (C + 6 digits) | String |
| gender | Giới tính (Male/Female) | Categorical |
| age | Tuổi khách hàng | Numeric |
| category | Danh mục sản phẩm | Categorical |
| quantity | Số lượng | Numeric |
| price | Đơn giá | Numeric |
| payment_method | Phương thức thanh toán | Categorical |
| invoice_date | Ngày giao dịch | DateTime |
| shopping_mall | Tên trung tâm thương mại | Categorical |

---

##  Cấu trúc Project

```
kpdl-customer-shopping/
│
├── README.md                          # File này
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── raw/                          # Data gốc từ Kaggle
│   │   └── customer_shopping.csv
│   └── processed/                    # Data đã xử lý
│       └── cleaned_data.csv
│
├── 01_data_preprocessing/            # THÀNH VIÊN 1
│   ├── data_preprocessing.ipynb      # Main notebook
│   ├── data_cleaning.py              # Utility functions
│   ├── preprocessing_report.md       # Báo cáo chi tiết
│   ├── label_encoders.pkl            # Saved encoders
│   └── figures/                      # Visualizations
│       ├── 01_missing_values.png
│       └── 02_outliers_detection.png
│
├── 02_eda_clustering/                # THÀNH VIÊN 2
│   ├── eda_analysis.ipynb            # EDA notebook
│   ├── clustering_analysis.ipynb     # Clustering notebook
│   ├── clustering_models.py          # Model implementations
│   ├── clustering_results.csv        # Results
│   ├── eda_clustering_report.md      # Báo cáo chi tiết
│   └── figures/                      # Visualizations
│       ├── 01_univariate_analysis.png
│       ├── 05_elbow_method.png
│       ├── 06_kmeans_clusters.png
│       └── ...
│
├── 03_association_rules/             # THÀNH VIÊN 3
│   ├── association_rules.ipynb       # Main notebook
│   ├── market_basket_analysis.py     # Apriori/FP-Growth code
│   ├── association_rules_results.csv # All rules
│   ├── frequent_itemsets.csv         # Frequent patterns
│   ├── actionable_insights.csv       # Business insights
│   ├── association_rules_report.md   # Báo cáo chi tiết
│   └── figures/                      # Visualizations
│       ├── 01_item_frequency.png
│       ├── 03_support_confidence_plot.png
│       ├── 04_rules_network.png
│       └── ...
│
├── 04_classification_integration/    # THÀNH VIÊN 4
│   ├── classification_models.ipynb   # Main notebook
│   ├── model_comparison.py           # Comparison utilities
│   ├── final_report.ipynb            # Final integration
│   ├── model_decision_tree.pkl       # Saved models
│   ├── model_random_forest.pkl
│   ├── model_naive_bayes.pkl
│   ├── model_comparison_results.csv  # Results
│   ├── classification_report.md      # Báo cáo chi tiết
│   └── figures/                      # Visualizations
│       ├── 01_dt_feature_importance.png
│       ├── 02_decision_tree_viz.png
│       ├── 07_model_comparison.png
│       └── ...
│
├── final_report/                     # Báo cáo cuối cùng
│   ├── final_report.pdf              # Báo cáo hoàn chỉnh (5 chương)
│   ├── presentation.pptx             # Slide thuyết trình
│   ├── summary.md                    # Tóm tắt insights
│   └── final_summary_report.txt      # Auto-generated summary
│
└── utils/                            # Shared utilities
    ├── data_loader.py                # Load data functions
    ├── visualization.py              # Plot functions
    └── evaluation_metrics.py         # Metric calculations
```

## Nhóm thực hiện

| Thành viên | Vai trò | Phần đảm nhận |
|------------|---------|---------------|
| **Thành viên 1** | Data Engineer | Data Preprocessing & Feature Engineering |
| **Thành viên 2** | Data Analyst | EDA & Clustering Analysis |
| **Thành viên 3** | Data Miner | Association Rules Mining |
| **Thành viên 4** | ML Engineer | Classification & Integration |

---

## Tài liệu tham khảo

1. Dataset: [Customer Shopping Dataset - Kaggle](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset/data)
2. Scikit-learn Documentation
3. MLxtend Documentation (Association Rules)
4. Data Mining: Concepts and Techniques (Han, Kamber, Pei)

---

## Acknowledgments

- Dataset author: Mehmet Tahir Arslan
- Kaggle Community
- Course instructors and TAs

---

*Last updated: December 2024*
