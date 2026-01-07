"""
HƯỚNG DẪN SỬ DỤNG NHANH
========================

1. CHUẨN BỊ & HUẤN LUYỆN
   python main.py --prepare_data --epochs 10

2. DỊCH CÂU (KHUYẾN NGHỊ - BEAM SEARCH)
   python generate.py
   hoặc
   python generate.py --prompt "Hello, how are you?"

3. DỊCH NHANH (GREEDY)
   python generate.py --prompt "Hello!" --greedy

4. TEST MODEL
   python test_model.py

5. ĐÁNH GIÁ
   python evaluate.py --model_path best_model.pt --dataset test

========================
CÁC LỖI THƯỜNG GẶP:
========================

❌ Lỗi: FileNotFoundError: best_model.pt
✅ Giải pháp: Chạy huấn luyện trước
   python main.py --prepare_data --epochs 10

❌ Lỗi: FileNotFoundError: tokenizer.model
✅ Giải pháp: Chạy prepare data
   python prepare_data.py

❌ Kết quả dịch không tốt
✅ Giải pháp:
   - Huấn luyện thêm epochs (20-30)
   - Dùng beam search: python generate.py --beam_size 10
   - Kiểm tra val_loss có < 3.0 không

❌ Out of Memory
✅ Giải pháp:
   python main.py --batch_size 16 --max_seq_length 64

========================
THAM SỐ QUAN TRỌNG:
========================

Huấn luyện:
  --epochs 10              # Số epochs (càng nhiều càng tốt, nhưng tốn thời gian)
  --batch_size 32          # Giảm nếu hết RAM/VRAM
  --lr 0.0001              # Learning rate
  --d_model 512            # Tăng để model mạnh hơn (512/768/1024)
  
Dịch:
  --beam_size 5            # Càng lớn càng tốt nhưng càng chậm (1-10)
  --greedy                 # Dùng greedy thay beam (nhanh nhưng chất lượng thấp)
  --max_tokens 100         # Độ dài câu dịch tối đa

========================
"""

print(__doc__)
