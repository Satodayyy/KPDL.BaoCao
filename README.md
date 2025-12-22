# Customer Shopping Data Mining Project

**Đề tài Khai phá Dữ liệu - Data Mining Course**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Cấu trúc Project](#cấu-trúc-project)
- [Nhóm thực hiện](#nhóm-thực-hiện)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
---

##  Giới thiệu

Dự án này thực hiện phân tích toàn diện dữ liệu mua sắm từ 10 trung tâm thương mại tại Istanbul, áp dụng các kỹ thuật khai phá dữ liệu bao gồm:

- **Clustering**: Phân nhóm khách hàng dựa trên RFM (Recency, Frequency, Monetary)
- **Association Rules**: Tìm kiếm mối quan hệ giữa các sản phẩm (Market Basket Analysis)
- **Classification**: Dự đoán danh mục sản phẩm dựa trên đặc điểm khách hàng

### Mục tiêu

1. Khám phá các mẫu (patterns) ẩn trong hành vi mua sắm của khách hàng.
2. Phân nhóm khách hàng nhằm hỗ trợ cá nhân hóa chiến lược marketing.
3. Xác định các tổ hợp sản phẩm thường được mua cùng nhau để phục vụ cross-selling.
4. Dự đoán xu hướng mua sắm của khách hàng dựa trên các đặc điểm nhân khẩu học.

### Câu hỏi nghiên cứu

1. Phân nhóm khách hàng (Customer Segmentation):
Có những nhóm khách hàng nào với hành vi mua sắm tương đồng? Các nhóm này được đặc trưng bởi những yếu tố gì về tần suất mua hàng, giá trị giao dịch và thời gian mua sắm gần nhất?
2. Khai phá luật kết hợp (Association Rules Discovery):
Những sản phẩm nào thường được mua cùng nhau? Mối quan hệ giữa các danh mục sản phẩm có đủ mạnh để áp dụng cho chiến lược cross-selling không?
3. Dự đoán hành vi mua sắm (Purchase Behavior Prediction):
Có thể dự đoán chính xác danh mục sản phẩm mà khách hàng sẽ mua dựa trên các đặc điểm nhân khẩu học, phương thức thanh toán, và thời điểm mua sắm không?
4. Phân tích xu hướng thời gian (Temporal Pattern Analysis):
Xu hướng mua sắm thay đổi như thế nào theo thời gian? Có sự khác biệt đáng kể giữa các ngày trong tuần, giờ trong ngày, hoặc các giai đoạn khác nhau trong năm không?
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
├── 01_data_preprocessing/            # Huỳnh Nhật Thành
│   ├── data_preprocessing.ipynb      # Main notebook             
│   ├── preprocessing_report.md       # Báo cáo chi tiết
│   ├── label_encoders.pkl            # Saved encoders
│   └── figures/                      # Visualizations
│       └── 02_outliers_detection.png
│
├── 02_eda_clustering/                # Lê Thành Danh
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
├── 03_association_rules/             # Nguyễn Quốc Thuận
│   ├── association_rules.ipynb       # Main notebook
│   ├── market_basket_analysis.ipynb  # Apriori/FP-Growth code
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
├── 04_classification_integration/    # Nguyễn Trí Sự
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
    ├── final_report.pdf              # Báo cáo hoàn chỉnh (5 chương)
    ├── presentation.pptx             # Slide thuyết trình
    ├── summary.md                    # Tóm tắt insights
    └── final_summary_report.txt      # Auto-generated summary

```

## Nhóm thực hiện

| Thành viên | Vai trò | Phần đảm nhận |
|------------|---------|---------------|
| **Huỳnh Nhật Thành** | Data Engineer | Data Preprocessing & Feature Engineering |
| **Lê Thành Danh** | Data Analyst | EDA & Clustering Analysis |
| **Nguyễn Quốc Thuận** | Data Miner | Association Rules Mining |
| **Nguyễn Trí Sự** | ML Engineer | Classification & Integration |

---

## Tài liệu tham khảo

1. Customer Shopping Dataset – Kaggle.  
   https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset/data

2. Scikit-learn Documentation.  
   https://scikit-learn.org/stable/

3. MLxtend Documentation – Association Rules.  
   https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

4. J. Han, M. Kamber, J. Pei, *Data Mining: Concepts and Techniques*, 3rd Edition, Elsevier, 2011.  
   https://www.sciencedirect.com/book/9780123814791/data-mining-concepts-and-techniques

5. S. A. Pratama et al., “Customer Segmentation Using RFM Model and K-Means Clustering,”  
   *INSTINK: Inovasi Teknik Informatika*, vol. 7, no. 2, 2022.  
   https://journal.unuha.ac.id/index.php/Instink/article/view/4349/1113


---

## Lời cảm ơn

- Tác giả bộ dữ liệu: Mehmet Tahir Arslan
- Cộng đồng Kaggle
- Giảng viên hướng dẫn: Thầy Đỗ Như Tài


---

*Last updated: December 2025*
