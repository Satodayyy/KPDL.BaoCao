
KẾT QUẢ PREPROCESSING:

1. DỮ LIỆU GỐC:
   - Số dòng: 99,457
   - Số cột: 10

2. DỮ LIỆU SAU XỬ LÝ:
   - Số dòng: 99,457 (100.00%)
   - Số cột: 29

3. CÁC BƯỚC ĐÃ THỰC HIỆN:
   - Xử lý missing values
   - Loại bỏ duplicates
   - Phát hiện và xử lý outliers
   - Feature engineering (total_amount, time features, age_group, RFM)
   - Chuẩn hóa dữ liệu (StandardScaler)
   - Label encoding cho categorical features

4. FEATURES MỚI:
   - total_amount
   - year, month, day, day_of_week, hour, quarter
   - age_group
   - recency, frequency, monetary (RFM)
   - price_range
   - is_weekend

5. SẴN SÀNG CHO:
   - Exploratory Data Analysis
   - Clustering Analysis
   - Association Rules Mining
   - Classification Models

OUTPUT FILES:
   - cleaned_data.csv
   - label_encoders.pkl
   - Figures: 02_outliers_detection.png
   - preprocessing_report.md
