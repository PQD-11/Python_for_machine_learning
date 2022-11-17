# Python_for_machine_learning
## BÀI TẬP: 
### Bài tập 1: Xử lý dữ liệu
Thực hiện các bước xử lý trên bộ dữ liệu iris, các bước phải có:
1. Tải dữ liệu
2. Chia dữ liệu huấn luyện và kiểm tra
3. Chuẩn hóa dữ liệu
4. Trực quan hóa dữ liệu
### Bài tập 2: Xử lý dữ liệu GIS
Bước 1: Cài đặt thư viện geopandas
Bước 2: git clone https://github.com/CityScope/CSL_HCMC
Bước 3: dùng geopandas để đọc shapefile trong /Data/GIS/Population/population_HCMC/population_shapefile/Population_Ward_Level.shp
Bước 4: hãy thực hiện các tác vụ truy vấn sau
- Phường nào có diện tích lớn nhất
- Phường nào có dân số 2019 (Pop_2019) cao nhất
- Phường nào có diện tích nhỏ nhất
- Phường nào có dân số thấp nhất (2019)
- Phường nào có tốc độ tăng trưởng dân số nhanh nhất (dựa trên Pop_2009 và Pop_2019)
- Phường nào có tốc độ tăng trưởng dân số thấp nhất
- Phường nào có biến động dân số nhanh nhất
- Phường nào có biến động dân số chậm nhất
- Phường nào có mật độ dân số cao nhất (2019)
- Phường nào có mật độ dân số thấp nhất (2019)
Khi làm trên notebook cần:
- Có MSSV và họ tên
- Ghi ra câu thông báo kết quả. Ví dụ: Phường có diện tích nhỏ nhất là phường 7 quận 4.
- Tải về file notebook và nộp chứ không nộp link
### Bài tập 3: Trực quan hóa dữ liệu GIS trên bản đồ
Bước 1: Cài đặt geopandas và folium
Bước 2: git clone https://github.com/CityScope/CSL_HCMC
Bước 3: dùng geopandas để đọc shapefile trong /Data/GIS/Population/population_HCMC/population_shapefile/Population_District_Level.shp
Bước 4: hãy thực hiện vẽ ranh giới các quận lên bản đồ dựa theo hướng dẫn sau:
https://geopandas.readthedocs.io/en/latest/gallery/polygon_plotting_with_folium.html
### Bài tập 4 - Gom cụm dữ liệu click của người dùng
Bước 1: Cài đặt các thư viện cần thiết
!pip install matplotlib==3.1.3
!pip install osmnet
!pip install folium
!pip install rtree
!pip install pygeos
!pip install geojson
!pip install geopandas
Bước 2: clone data từ https://github.com/CityScope/CSL_HCMC
Bước 3: Load ranh giới quận huyện và dân số quận huyện từ: Data\GIS\Population\population_HCMC\population_shapefile\Population_District_Level.shp
Bước 4: Load dữ liệu click của người dùng
Bước 5: Lọc ra 5 quận huyện có tốc độ tăng MẬT ĐỘ dân số nhanh nhất (Dùng dữ liệu 2019  và 2017)
Bước 6: Dùng spatial join (from geopandas.tools import sjoin) để lọc ra các điểm click của người dùng trong 5 quận/huyện hot nhất
Bước 7: chạy KMean cho top 5 quận huyện này. Lấy K = 20
Bước 8: Lưu 01 cụm điểm nhiều nhất trong các quận huyện ở Bước 5.
Bước 9: show lên bản đồ các cụm đông nhất theo từng quận huyện theo dạng HEATMAP
Bước 10: Lưu heatmap xuống file png
### Bài tập 5 - Linear Regression với Streamlit
Sử dụng Streamlit để làm giao diện ứng dụng theo gợi ý trên lớp lý thuyết.
Yêu cầu bao gồm:
Thiết kế giao diện với Streamlit để có thể:
- Upload file csv (sau này có thể thay bằng tập dữ liệu khác dễ dàng).
- Hiển thị bảng dữ liệu với file đã upload
- Chọn lựa input feature (các cột dữ liệu đầu vào)
- Chọn lựa hệ số cho train/test split: Ví dụ 0.8 có nghĩa là 80% để train và 20% để test
- Chọn lựa hệ số K cho K-Fold cross validation: Ví dụ K =4
- Nút "Run" để tiến hành chạy và đánh giá thuật toán
Output sẽ là biểu đồ cột hiển thị các kết quả sử dụng độ đo MAE và MSE. Lưu ý: Train/Test split và K-Fold cross validation được thực hiện độc lập, chỉ chọn 1 trong hai phương pháp này.
### Bài tập 6 - Logistic Regression với Streamlit
Sử dụng Streamlit để làm giao diện ứng dụng theo gợi ý trên lớp lý thuyết.

Yêu cầu bao gồm:
Thiết kế giao diện với Streamlit để có thể:
- Upload file csv (sau này có thể thay bằng tập dữ liệu khác dễ dàng).
- Hiển thị bảng dữ liệu với file đã upload
- Chọn lựa input feature (các cột dữ liệu đầu vào)
- Chọn lựa hệ số cho train/test split: Ví dụ 0.8 có nghĩa là 80% để train và 20% để test
- Chọn lựa hệ số K cho K-Fold cross validation: Ví dụ K =4
- Nút "Run" để tiến hành chạy và đánh giá thuật toán

Output sẽ là biểu đồ cột hiển thị các kết quả sử dụng độ đo Precision, Recall, F1 và Log Loss. Lưu ý: Train/Test split và K-Fold cross validation được thực hiện độc lập, chỉ chọn 1 trong hai phương pháp này.
