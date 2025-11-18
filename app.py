import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="Hệ Thống Gợi Ý Bài Hát", layout="wide")

# Hàm tải mô hình (Dùng cache để chỉ tải 1 lần)
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model_components = pickle.load(f)
        
        return (
            model_components['cosine_sim_matrix'],
            model_components['song_indices'],
            model_components['df_model']
        )
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}. Vui lòng kiểm tra file model.pkl.")
        return None, None, None

# Hàm Gợi Ý (Cần thay đổi để nhận ID/Index thay vì Title)
def get_recommendations_by_index(idx, cosine_sim_matrix, df_data, top_n=10):
    """
    Hàm gợi ý dựa trên chỉ mục (index) của bài hát.
    """
    
    # Lấy tất cả điểm tương đồng
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Bỏ bài hát đầu tiên (chính nó)
    sim_scores = sim_scores[1:top_n+1]
    
    song_indices = [i[0] for i in sim_scores]
    
    recommendations = df_data.iloc[song_indices].copy()
    recommendations['Điểm Tương Đồng'] = [f"{i[1]*100:.2f}%" for i in sim_scores]
    
    return recommendations[['Title', 'Artist', 'Genre', 'Điểm Tương Đồng']]


# --- Chức năng chính của Streamlit App ---
cosine_sim, indices, df_model = load_model()

if df_model is not None:
    st.title("🎶 Hệ Thống Gợi Ý Bài Hát (Content-Based) 🇻🇳")
    st.markdown("Chọn một bài hát để tìm kiếm 10 bài hát tương đồng nhất.")

    # 1. Tạo danh sách hiển thị: Tiêu đề - Nghệ sĩ
    display_list = (df_model['Title'] + ' - ' + df_model['Artist']).tolist()
    
    # Lấy danh sách chỉ mục (Index) tương ứng với display_list
    index_list = df_model.index.tolist()

    if not display_list:
        st.warning("Cơ sở dữ liệu bài hát trống.")
    else:
        # 2. Chọn bài hát đầu vào
        selected_display = st.selectbox(
            "1. Chọn Bài Hát Đầu Vào:",
            display_list,
            index=0 # Đảm bảo có giá trị mặc định
        )
        
        # 3. Lấy CHỈ MỤC (Index) của bài hát đã chọn
        # Vị trí của bài hát trong display_list
        selected_pos = display_list.index(selected_display)
        # Chỉ mục thực tế trong ma trận Cosine
        selected_index = index_list[selected_pos]
        
        st.write(f"Đang gợi ý cho bài hát: **{selected_display}** (Index: {selected_index})")

        if st.button("Tìm Gợi Ý"):
            with st.spinner('Đang tìm kiếm bài hát tương đồng...'):
                
                # GỌI HÀM GỢI Ý BẰNG CHỈ MỤC
                results = get_recommendations_by_index(
                    selected_index, 
                    cosine_sim, 
                    df_model, 
                    top_n=10
                )
                
                if results.empty:
                    st.warning(f"Không thể tìm thấy gợi ý cho '{selected_display}'.")
                else:
                    st.success("Top 10 bài hát tương đồng:")
                    # Thiết lập lại tên cột cho đẹp
                    results.columns = ['Tiêu Đề', 'Nghệ Sĩ', 'Thể Loại', 'Điểm Tương Đồng']
                    st.dataframe(results, use_container_width=True)