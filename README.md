# <div align="center"> Hệ Thống Gợi Ý Bài Hát Dựa Trên Nội Dung</div>
<div align="center"><em>Content-Based Song Recommendation System for Vietnamese Music</em></div>

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+"/>
    <img src="https://img.shields.io/badge/Pandas-1.5+-green.svg" alt="Pandas 1.5+"/>
    <img src="https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg" alt="Scikit-learn 1.0+"/>
    <img src="https://img.shields.io/badge/Streamlit-1.0+-red.svg" alt="Streamlit 1.0+"/>
    <br>
    <img src="https://img.shields.io/badge/Machine%20Learning-Recommendation-blue.svg" alt="Machine Learning - Recommendation"/>
    <img src="https://img.shields.io/badge/Music-Vietnamese%20Songs-yellow.svg" alt="Music - Vietnamese Songs"/>
    <img src="https://img.shields.io/badge/Status-Completed-brightgreen.svg" alt="Status - Completed"/>
</div>

## Tổng Quan

Dự án này tập trung vào việc xây dựng hệ thống gợi ý bài hát dựa trên nội dung (content-based recommendation system), sử dụng dữ liệu lời bài hát, thể loại, nghệ sĩ, và các yếu tố liên quan để khuyến nghị các bài hát tương đồng. Hệ thống nhắm đến cải thiện trải nghiệm người dùng trên các nền tảng nghe nhạc Việt Nam, nơi người dùng thường tìm kiếm bài hát dựa trên sở thích về lời bài hát, tâm trạng hoặc phong cách âm nhạc.

Ngành công nghiệp âm nhạc Việt Nam đang phát triển mạnh mẽ với sự phổ biến của các nền tảng streaming như Zing MP3, Spotify, và Apple Music. Gợi ý bài hát chính xác giúp tăng thời gian sử dụng ứng dụng, cải thiện mức độ hài lòng của người dùng, và tăng doanh thu từ quảng cáo hoặc đăng ký premium. Content-based filtering là lựa chọn phù hợp khi dữ liệu hành vi người dùng hạn chế, tập trung vào đặc trưng nội tại của bài hát như lời bài hát và thể loại.

Việc người dùng bỏ qua các gợi ý không phù hợp có thể làm giảm tương tác, trong khi gợi ý chính xác giúp tăng mức độ gắn bó. Dự án này hỗ trợ tạo playlist tự động và khuyến nghị cá nhân hóa, giúp người dùng khám phá các bài hát tương tự với sở thích hiện tại.

### Lợi ích kinh doanh:

- **Tăng tương tác người dùng**: Gợi ý bài hát phù hợp giúp tăng thời gian nghe nhạc lên 20-30%.
- **Giảm tỷ lệ bỏ qua**: Tăng tỷ lệ click vào bài hát gợi ý lên 15-25% nhờ độ tương đồng cao.
- **Tối ưu hóa playlist**: Tạo playlist tự động, giảm chi phí biên tập thủ công.
- **Tăng doanh thu**: Người dùng hài lòng dẫn đến tăng đăng ký premium và doanh thu quảng cáo.
- **Cải thiện trải nghiệm**: Gợi ý dựa trên tâm trạng hoặc chủ đề từ lời bài hát.
- **Phát triển cộng đồng**: Khuyến khích chia sẻ playlist, tăng mạng lưới người dùng.
- **Phân bổ tài nguyên hiệu quả**: Tập trung vào bài hát phổ biến hoặc mới với tiềm năng cao.

## Mục Tiêu

<div align="center">
  <table>
    <tr>
      <td align="center"><b>🔍</b></td>
      <td>Phân tích dữ liệu bài hát và xác định các yếu tố ảnh hưởng đến độ tương đồng</td>
    </tr>
    <tr>
      <td align="center"><b>🤖</b></td>
      <td>Xây dựng mô hình gợi ý dựa trên TF-IDF và cosine similarity</td>
    </tr>
    <tr>
      <td align="center"><b>💡</b></td>
      <td>Triển khai ứng dụng Streamlit để gợi ý bài hát và đề xuất cải tiến hệ thống</td>
    </tr>
  </table>
</div>

### Chi tiết mục tiêu:

1. **Phân tích dữ liệu bài hát**:
   - Khám phá các đặc trưng như lời bài hát (Lyrics_Keywords), thể loại, nghệ sĩ, và tags.
   - Xác định các yếu tố chính ảnh hưởng đến độ tương đồng (ví dụ: từ khóa chung trong lyrics).
   - Phân đoạn bài hát dựa trên thể loại và phong cách âm nhạc.
   - Tính toán TF-IDF để đánh giá trọng số từ vựng trong trường Content.

2. **Xây dựng mô hình gợi ý**:
   - Phát triển mô hình content-based với độ chính xác cao (cosine similarity > 0.8 cho top matches).
   - Tối ưu hóa TF-IDF với stop words tiếng Việt và giới hạn từ vựng (max_features=1000).
   - Giảm kích thước dữ liệu xuống 2.000 bài để tránh vấn đề bộ nhớ.
   - Đảm bảo top 10 gợi ý có similarity score cao và hiển thị thông tin rõ ràng (Title, Artist, Genre).

3. **Triển khai và đề xuất giải pháp**:
   - Triển khai ứng dụng Streamlit cho phép người dùng chọn bài hát và xem gợi ý.
   - Đề xuất chiến lược tích hợp với API nhạc để phát bài hát trực tiếp.
   - Xây dựng quy trình cập nhật dữ liệu bài hát mới.
   - Đề xuất cải tiến như hybrid model (kết hợp user-based filtering).

## Định Nghĩa Tương Đồng

Trong dự án này, chúng ta định nghĩa "Tương Đồng" (Similarity) như sau:

<div class="slide-body" style="margin-left: 20px; margin-bottom: 15px;">
  <ul>
    <li>📝 <b>Dựa trên nội dung văn bản</b>: Kết hợp các trường Title, Artist, Genre, SubGenre, Tags, và Lyrics_Keywords.</li>
    <li>📊 <b>Đo lường bằng cosine similarity</b>: Giá trị từ 0-1, với >0.8 là tương đồng cao.</li>
    <li>🔍 <b>Top N gợi ý</b>: Lấy 10 bài hát có similarity score cao nhất, loại trừ bài hát gốc.</li>
    <li>⚙️ <b>Xử lý tiếng Việt</b>: Loại bỏ stop words tiếng Việt và giới hạn từ vựng để tối ưu TF-IDF.</li>
  </ul>
</div>

### Giải thích chi tiết:

- **Nội dung văn bản**: Trường 'Content' được tạo bằng cách nối các trường Title, Artist, Genre, SubGenre, Tags, và Lyrics_Keywords sau khi làm sạch bằng regex để loại bỏ ký tự đặc biệt.
- **Cosine similarity**: Đo lường góc giữa các vector TF-IDF, tập trung vào hướng nội dung, không phụ thuộc độ dài văn bản.
- **Top N gợi ý**: Mặc định top 10, sắp xếp theo similarity score giảm dần, hiển thị dưới dạng phần trăm (%).
- **Xử lý tiếng Việt**: Sử dụng danh sách stop words tiếng Việt (như 'là', 'và', 'của') để loại bỏ từ vô nghĩa, và giới hạn max_features=1000 để giảm nhiễu và tiết kiệm bộ nhớ.
## Dữ Liệu
| Cột              | Ý nghĩa                                                                 | Vai trò trong Mô hình Gợi ý                                                                                   |
|------------------|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| SongID           | Mã định danh duy nhất của bài hát.                                     | Định danh: Dùng để theo dõi và ánh xạ kết quả gợi ý.                                                          |
| Title            | Tên bài hát.                                                           | Định danh & Nội dung: Cung cấp thông tin quan trọng. Được dùng làm đầu vào cho người dùng và là một phần của trường Content. |
| Artist           | Tên nghệ sĩ/ca sĩ trình bày.                                           | Nội dung Quan trọng: Là đặc trưng có trọng số cao nhất . Cần thiết để gợi ý các bài hát cùng ca sĩ. |
| Genre / SubGenre | Thể loại chính và thể loại phụ của bài hát (ví dụ: Ballad, V-Pop / Ballad Acoustic). | Nội dung Quan trọng: Phân loại âm nhạc. Thể loại cũng được tăng trọng số trong trường Content để ưu tiên gợi ý các bài hát tương đồng về phong cách. |
| Album / ReleaseYear | Tên Album và Năm phát hành.                                         | Bối cảnh: Mặc dù không được sử dụng trực tiếp trong ma trận Cosine Similarity, chúng cung cấp bối cảnh lịch sử và có thể được dùng cho các mô hình lai (Hybrid) hoặc để lọc kết quả. |
| Popularity       | Mức độ phổ biến của bài hát (dữ liệu số).                              | Đặc trưng Số: Được chuẩn hóa bằng MinMaxScaler . Có thể được dùng để sắp xếp lại kết quả gợi ý (ví dụ: chỉ gợi ý các bài hát tương đồng mà có độ phổ biến cao). |
| Duration_ms      | Thời lượng bài hát tính bằng mili-giây.                                | Đặc trưng Số (Phụ): Có thể dùng cho mục đích phân tích nhưng ít tác động đến mô hình Content-Based.           |
| Tags             | Các thẻ mô tả cảm xúc, chủ đề, hoặc bối cảnh sử dụng (ví dụ: \"quán cà phê\", \"tâm trạng\", \"remix\"). | Nội dung Ngữ cảnh: Rất quan trọng! Giúp gợi ý các bài hát có cảm xúc và mục đích sử dụng tương đồng, nâng cao chất lượng gợi ý. |
| Lyrics_Keywords  | Các từ khóa quan trọng được trích xuất từ lời bài hát (ví dụ: \"mưa\", \"chia tay\", \"tình yêu\"). | Nội dung Sâu: Cung cấp thông tin về chủ đề cốt lõi của bài hát, là một thành phần thiết yếu trong trường Content để tìm ra sự tương đồng về chủ đề. |
## Nội Dung

Dự án bao gồm các notebook và file sau:

<div style="background-color: #1068c0ff; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
  <ol>
    <li><b>1_Data_Generation.ipynb</b>: Placeholder cho thu thập dữ liệu (giả định dữ liệu gốc từ data.csv).</li>
    <li><b>2_Exploratory_Data_Analysis.ipynb</b>: Phân tích dữ liệu khám phá (EDA) và chuẩn hóa dữ liệu.</li>
    <li><b>3_Model_Building.ipynb</b>: Xây dựng mô hình TF-IDF và cosine similarity.</li>
    <li><b>4_Model_Evaluation.ipynb</b>: Đánh giá mô hình và kiểm tra gợi ý với bài hát mẫu.</li>
    <li><b>app.py</b>: Ứng dụng Streamlit để triển khai hệ thống gợi ý.</li>
  </ol>
</div>

### Chi tiết nội dung từng file:

#### 1_Data_Generation.ipynb
- Placeholder cho việc thu thập dữ liệu bài hát từ các nguồn như Zing MP3, Spotify, hoặc API nhạc.
- Không chứa code thực thi, giả định dữ liệu gốc được cung cấp trong `data.csv`.

#### 2_Exploratory_Data_Analysis.ipynb
- Tải dữ liệu từ `data.csv` (50.000 bài hát, 11 cột).
- Làm sạch và tạo trường 'Content' bằng cách kết hợp Title, Artist, Genre, SubGenre, Tags, Lyrics_Keywords.
- Chuẩn hóa Popularity bằng MinMaxScaler, tạo trường Popularity_Normalized.
- Vẽ histogram phân bố Popularity để kiểm tra dữ liệu.
- Lưu dữ liệu đã xử lý vào `data_EDA.csv` với 7 cột cần thiết.

#### 3_Model_Building.ipynb
- Tải `data_EDA.csv`, giảm kích thước xuống 2.000 bài để tối ưu bộ nhớ.
- Khởi tạo TfidfVectorizer với stop words tiếng Việt, max_features=1000, max_df=0.8, min_df=2.
- Tính ma trận TF-IDF (2000x141) và ma trận cosine similarity (2000x2000, float32).
- Tạo ánh xạ Title-Index và lưu các thành phần (tfidf, cosine_sim, indices, df_model) vào `model.pkl`.

#### 4_Model_Evaluation.ipynb
- Tải `model.pkl` và định nghĩa hàm `get_recommendations` dựa trên tiêu đề bài hát.
- Kiểm tra với bài hát mẫu (ví dụ: "Xa Em Buồn - Yvonne"), in top 10 bài hát tương đồng.
- Hiển thị kết quả với Title, Artist, Genre, và Similarity_Score.

#### 5. app.py
- Triển khai ứng dụng Streamlit với giao diện chọn bài hát qua selectbox.
- Sử dụng hàm `get_recommendations_by_index` để gợi ý dựa trên index, tránh lỗi trùng tiêu đề.
- Hiển thị top 10 bài hát tương đồng trong bảng với các cột Tiêu Đề, Nghệ Sĩ, Thể Loại, Điểm Tương Đồng.

## Các Yếu Tố Ảnh Hưởng

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; margin-bottom: 20px;">
  <div style="flex: 0 0 48%; background-color: #06005eff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <h4> Nội dung văn bản</h4>
    <ul>
      <li>Lời bài hát (Lyrics_Keywords)</li>
      <li>Tiêu đề bài hát</li>
      <li>Tags (ví dụ: tâm trạng, remix)</li>
    </ul>
  </div>
  <div style="flex: 0 0 48%; background-color: #1cb16bff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <h4> Thông tin bài hát</h4>
    <ul>
      <li>Nghệ sĩ</li>
      <li>Thể loại (Genre, SubGenre)</li>
      <li>Popularity</li>
    </ul>
  </div>
</div>

## Các Mô Hình Sử Dụng

<table align="center">
  <tr>
    <th>Mô Hình</th>
    <th>Ưu Điểm</th>
    <th>Nhược Điểm</th>
  </tr>
  <tr>
    <td>TF-IDF + Cosine Similarity</td>
    <td>Dễ triển khai, hiệu quả với văn bản, không cần dữ liệu người dùng</td>
    <td>Chỉ dựa trên nội dung, không cá nhân hóa theo hành vi</td>
  </tr>
</table>

### Chi tiết về cách tiếp cận:

#### 1. TF-IDF + Cosine Similarity
TF-IDF (Term Frequency-Inverse Document Frequency) được sử dụng để chuyển đổi trường 'Content' thành vector, với trọng số dựa trên tần suất và độ hiếm của từ. Cosine similarity đo lường độ tương đồng giữa các vector, ưu tiên hướng nội dung.

```python
# Ví dụ code triển khai TF-IDF và Cosine Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words=VIETNAMESE_STOP_WORDS, max_features=1000, max_df=0.8, min_df=2)
tfidf_matrix = tfidf.fit_transform(df_model['Content'])
cosine_sim = cosine_similarity(tfidf_matrix).astype(np.float32)
```

## Đánh Giá Mô Hình

<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <div style="flex: 0 0 48%; background-color: #690707ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <h4> Các Metrics Đánh Giá</h4>
    <ul>
      <li>Cosine Similarity Score</li>
      <li>Qualitative Evaluation (Top 10 gợi ý)</li>
    </ul>
  </div>
  <div style="flex: 0 0 48%; background-color: #630404ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <h4> Phương Pháp Kiểm Tra</h4>
    <ul>
      <li>Test với bài hát mẫu</li>
      <li>So sánh nghệ sĩ và thể loại của gợi ý</li>
      <li>Đánh giá độ tương đồng trên Streamlit</li>
    </ul>
  </div>
</div>

### Kết quả đánh giá mô hình:

<table align="center">
  <tr>
    <th>Bài Hát Mẫu</th>
    <th>Top 1 Similarity</th>
    <th>Top 5 Similarity</th>
    <th>Top 10 Similarity</th>
  </tr>
  <tr>
    <td>Xa Em Buồn - Yvonne</td>
    <td>0.884018</td>
    <td>0.821651-0.884018</td>
    <td>0.821651-0.884018</td>
  </tr>
</table>

### Phân tích feature importance:

<div style="background-color: #610731ff; padding: 15px; border-radius: 5px; margin: 15px 0;">
  <p>Các yếu tố có ảnh hưởng lớn nhất đến độ tương đồng:</p>
  <ol>
    <li><b>Lời bài hát (Lyrics_Keywords)</b>: Từ khóa như "mưa", "nắng", "chia tay" quyết định sự tương đồng trong ballad.</li>
    <li><b>Nghệ sĩ</b>: Bài hát cùng nghệ sĩ (như Trúc Nhân) có similarity cao.</li>
    <li><b>Thể loại</b>: Ballad với ballad, Pop với Pop có similarity cao.</li>
    <li><b>Tags</b>: Tags như "tâm trạng", "remix" ảnh hưởng đến vector TF-IDF.</li>
    <li><b>Tiêu đề</b>: Từ trong tiêu đề đóng góp vào độ tương đồng.</li>
  </ol>
</div>

### Phân tích các phân đoạn bài hát:

- **Ballad**: Nhóm lớn nhất, có similarity cao nhờ từ ngữ cảm xúc trong lyrics.
- **Pop/Rock**: Similarity dựa trên nhịp điệu hoặc mô tả trong tags.
- **Bài hát mới**: Có thể có similarity thấp hơn nếu từ khóa ít.
- **Bài hát phổ biến**: Thường có similarity cao với nhiều bài cùng thể loại/nghệ sĩ.

## Cấu Trúc Dự Án

```
Song_Recommendation_System/

 data/                             # Thư mục chứa dữ liệu
    data.csv                      # Dữ liệu gốc (50.000 bài hát)
    data_EDA.csv                  # Dữ liệu sau EDA
    
 models/
    models.pkl                       # Mô hình đã lưu

 notebooks/                        # Jupyter notebooks
    1_Data_Generation.ipynb                   # Placeholder thu thập dữ liệu
    2_Exploratory_Data_Analysis.ipynb                   # EDA và chuẩn hóa dữ liệu
    3_Model_Building.ipynb                   # Xây dựng mô hình
    4_Model_Evaluation.ipynb                   # Đánh giá mô hình

 app.py                            # Ứng dụng Streamlit

 README.md                         # Tài liệu dự án này
```

## Đề Xuất Giải Pháp

Dựa trên kết quả phân tích và mô hình gợi ý, chúng tôi đề xuất các giải pháp sau để cải thiện hệ thống và tăng trải nghiệm người dùng:

### 1. Chiến lược theo phân khúc

<div style="background-color: #04834eff; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
  <h4>📱 Người dùng trẻ (Ballad/Pop)</h4>
  <ul>
    <li>Tích hợp gợi ý vào ứng dụng di động với playlist theo tâm trạng (buồn, vui, lãng mạn).</li>
    <li>Cung cấp ưu đãi nghe miễn phí cho bài hát tương đồng.</li>
    <li>Tích hợp tính năng chia sẻ playlist qua mạng xã hội.</li>
    <li>Sử dụng AI để gợi ý dựa trên thời gian nghe (sáng/tối).</li>
  </ul>
</div>

<div style="background-color: #7706a3ff; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
  <h4>💰 Người dùng nghe ít (thấp similarity)</h4>
  <ul>
    <li>Gửi push notification nhắc nhở với gợi ý bài hát mới.</li>
    <li>Chương trình thử nghe miễn phí top bài hát gợi ý.</li>
    <li>Hoàn tiền premium nếu không hài lòng với gợi ý.</li>
    <li>Khảo sát feedback để cải thiện mô hình gợi ý.</li>
  </ul>
</div>

<div style="background-color: #0caa0cff; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
  <h4>🔄 Người dùng nghe thường xuyên</h4>
  <ul>
    <li>Tạo playlist cá nhân hóa hàng tuần dựa trên gợi ý.</li>
    <li>Tích điểm thưởng khi nghe bài hát từ gợi ý.</li>
    <li>Ưu đãi vé concert cho nghệ sĩ trong danh sách gợi ý.</li>
    <li>Dịch vụ premium với gợi ý không giới hạn.</li>
  </ul>
</div>

<div style="background-color: #1c0291ff; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
  <h4>👑 Người dùng VIP (nhiều lượt nghe)</h4>
  <ul>
    <li>Gợi ý độc quyền bài hát mới từ nghệ sĩ yêu thích.</li>
    <li>Tích hợp với thiết bị thông minh (smart speaker) để phát nhạc.</li>
    <li>Ưu đãi đặc biệt như meet & greet nghệ sĩ.</li>
    <li>Cung cấp báo cáo phân tích sở thích nghe nhạc cá nhân.</li>
  </ul>
</div>

### 2. Chiến lược theo mức độ similarity

- **Nhóm similarity rất cao (>80%)**:
  - Tự động thêm vào playlist yêu thích.
  - Gửi thông báo "Khám phá bài hát tương tự bạn yêu thích".

- **Nhóm similarity cao (60-80%)**:
  - Gợi ý trong mục "Khám phá mới" trên ứng dụng.
  - Kết hợp với bài hát trending để tăng tính mới mẻ.

- **Nhóm similarity trung bình (40-60%)**:
  - Sử dụng để đa dạng hóa playlist, tránh lặp lại.
  - Kết hợp với feedback người dùng để tinh chỉnh mô hình.

### 3. Cải tiến sản phẩm và dịch vụ

- **Ứng dụng di động**: Nâng cấp giao diện với carousel gợi ý, hỗ trợ tìm kiếm bằng giọng nói.
- **Tích hợp API nhạc**: Kết nối với Zing MP3/Spotify để phát bài hát trực tiếp từ gợi ý.
- **Hybrid model**: Kết hợp content-based với collaborative filtering khi có dữ liệu người dùng.
- **Cập nhật dữ liệu**: Xây dựng pipeline tự động crawl bài hát mới hàng tuần để cập nhật model.

## Kết Luận và Hướng Phát Triển

Dự án đã thành công trong việc xây dựng hệ thống gợi ý bài hát dựa trên nội dung, sử dụng TF-IDF và cosine similarity, với triển khai thực tế qua ứng dụng Streamlit. Mô hình xử lý tốt dữ liệu tiếng Việt, cho ra gợi ý có độ tương đồng cao (trên 0.8 cho top matches), đặc biệt với các bài hát cùng nghệ sĩ hoặc thể loại.

### Hướng phát triển trong tương lai:

1. **Hybrid recommendation**: Kết hợp content-based với collaborative filtering để cá nhân hóa tốt hơn.
2. **Phân tích sentiment lyrics**: Sử dụng NLP (như PhoBERT) để gợi ý dựa trên cảm xúc của lời bài hát.
3. **Tự động tạo playlist**: Phát triển thuật toán tạo playlist theo chủ đề hoặc tâm trạng.
4. **Cập nhật real-time**: Tích hợp API để thêm bài hát mới vào hệ thống.
5. **Tích hợp dữ liệu thực**: Áp dụng mô hình với dữ liệu từ nền tảng streaming thực tế và đánh giá hiệu quả.

<div align="center">
  <p>Done.</p>
</div>