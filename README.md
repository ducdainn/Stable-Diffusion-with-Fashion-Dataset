# Dự án Tinh chỉnh Stable Diffusion với Dữ liệu Thời trang

Dự án này trình bày quy trình xử lý dữ liệu ảnh và caption, tinh chỉnh mô hình Stable Diffusion sử dụng LoRA (Low-Rank Adaptation) trên tập dữ liệu thời trang, và sử dụng mô hình đã tinh chỉnh để tạo ảnh mới.

## Hướng dẫn

### 1. Tải Dữ liệu

Để bắt đầu, bạn cần tải tập dữ liệu ảnh và caption.
*   **Tập hợp ảnh**: Tải từ liên kết sau: [Images](https://drive.google.com/file/d/1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN/view?usp=sharing).
*   **File caption**: Tải file JSON chứa caption từ liên kết sau: [Captions](https://drive.google.com/file/d/1d1TRm8UMcQhZCb6HpPo8l3OPEin4Ztk2/view?usp=sharing).

Sau khi tải về, bạn sẽ có các tệp cần thiết cho các bước tiếp theo.

### 2. Xử lý Dữ liệu (Data Processing)

Bước này chuẩn bị dữ liệu thô thành định dạng phù hợp cho quá trình training.
*   Sử dụng file `images_processing.ipynb` để thực hiện quá trình lọc và chuẩn bị dữ liệu.
*   Script này có khả năng lọc ảnh dựa trên độ phân giải tối thiểu (mặc định 256x256) và độ dài caption tối thiểu (mặc định 10 ký tự).
*   Quá trình xử lý bao gồm việc đọc file caption, kiểm tra tính hợp lệ của ảnh (tồn tại, định dạng, độ phân giải, kênh màu), chọn ngẫu nhiên một số lượng ảnh nhất định (mặc định 10,000 ảnh), tạo DataFrame chứa thông tin ảnh và caption đã lọc, sao chép các ảnh đã chọn sang thư mục đầu ra, và tùy chọn tạo file ZIP chứa ảnh đã xử lý.
*   File output chính từ bước này là `fashion_data.csv`.

Bạn có thể chạy script này (hoặc notebook tương ứng) để chuẩn bị dữ liệu.

### 3. Huấn luyện Mô hình (Training)

Quy trình huấn luyện mô hình Stable Diffusion được thực hiện bằng cách tinh chỉnh (fine-tuning) mô hình cơ sở sử dụng kỹ thuật LoRA.
*   Sử dụng file `stable-diffusion.ipynb` cho quá trình training.
*   **Yêu cầu hệ thống**: Cần chuyển sang chế độ sử dụng GPU (ví dụ: trên Kaggle) để chạy toàn bộ code. Script kiểm tra các yêu cầu hệ thống như tính khả dụng của CUDA, dung lượng GPU, và sự tồn tại của các thư mục cần thiết.
*   **Cấu hình**: Các tham số cấu hình quan trọng được định nghĩa trong lớp `Config`, bao gồm đường dẫn dữ liệu, hyperparameters training (kích thước batch, learning rate, số bước, v.v.), cấu hình LoRA (rank, alpha, dropout), và các tham số cho validation/early stopping.
*   **Tiền xử lý ảnh**: Ảnh được tải, tăng cường chất lượng (tăng độ sắc nét, tương phản, màu sắc nhẹ) và thay đổi kích thước về 512x512. Một lớp `AdvancedImageProcessor` được sử dụng để kiểm tra chất lượng ảnh (kích thước, độ sắc nét, tỷ lệ khung hình) và chỉ xử lý các ảnh đạt tiêu chí.
*   **Precompute Latents**: Thay vì xử lý ảnh trong lúc training, script tính toán và lưu trữ biểu diễn latent của ảnh trước bằng VAE encoder của Stable Diffusion để tăng tốc độ training. Latents được lưu dưới dạng các file `.pt`.
*   **Dataset và DataLoader**: Dữ liệu được tải từ file CSV đã xử lý. Một lớp `EnhancedFashionDataset` tùy chỉnh được sử dụng, bổ sung các kỹ thuật tăng cường dữ liệu trực tiếp trên không gian latent (`LatentAugmentation`) như thêm nhiễu, lật ngang, và cắt/thay đổi kích thước ngẫu nhiên. Caption cũng được tăng cường với các thuật ngữ chất lượng dành riêng cho thời trang trong quá trình training. Dataset được chia thành tập train và validation (mặc định 92% train, 8% val).
*   **Mô hình**: Mô hình Stable Diffusion v1-5 từ `runwayml` được sử dụng làm mô hình cơ sở. Scheduler được cấu hình.
*   **LoRA**: Kỹ thuật LoRA được áp dụng lên các module cụ thể của UNet (to_k, to_q, to_v, to_out.0, proj_in, proj_out, conv1, conv2) với các tham số `lora_rank`, `lora_alpha`, `lora_dropout` được cấu hình (mặc định rank 12, alpha 24, dropout 0.1).
*   **Optimizer và Scheduler**: Sử dụng AdamW optimizer và Cosine Annealing Warm Restarts scheduler với warmup steps. Mixed precision (`autocast`) và Gradient Scaler được dùng để tối ưu hóa việc sử dụng GPU.
*   **Vòng lặp Training**: Mô hình được huấn luyện trong số bước đã cấu hình (mặc định 2000 bước). Loss (MSE) được tính toán giữa nhiễu dự đoán và nhiễu thực tế. Gradient clipping được áp dụng.
*   **Validation và Early Stopping**: Quá trình validation được thực hiện định kỳ (mặc định sau mỗi 500 bước) trên một phần nhỏ tập validation. Các metrics validation như Val Loss (MSE), MAE, Loss Std được tính toán. Mô hình tốt nhất dựa trên Validation Loss được lưu lại. Early stopping được triển khai với tham số `patience` (mặc định 10) và `min_delta`.
*   **Theo dõi và Báo cáo**: Progress training được hiển thị qua TQDM và các thông số như loss, grad norm, learning rate, thời gian ước tính được in ra console. Một `ModelEvaluator` được sử dụng để theo dõi metrics và vẽ biểu đồ tiến trình training. Cuối cùng, một báo cáo training chi tiết được tạo và lưu dưới dạng file JSON.

### 4. Inference (Tạo ảnh)

Sau khi huấn luyện, bạn có thể sử dụng mô hình đã tinh chỉnh để tạo ra các ảnh thời trang mới từ caption văn bản.
*   Mô hình đã tinh chỉnh (LoRA weights) được tải từ thư mục lưu (`/kaggle/working/fashion_diffusion_best/best_model`). Script `safe_load_best_model` được cung cấp để thử các chiến lược tải khác nhau. Cần lưu ý các cảnh báo khi tải LoRA nếu cấu hình không khớp hoàn toàn.
*   Hàm `generate_fashion_images` được sử dụng để tạo ảnh.
*   **Prompts**: Bạn cung cấp các câu lệnh (prompts) mô tả ảnh thời trang muốn tạo.
*   **Negative Prompts**: Một negative prompt được sử dụng để loại bỏ các yếu tố không mong muốn như ảnh bị mờ, méo mó, chất lượng thấp, các vấn đề về giải phẫu, nhiều người trong ảnh, v.v.. Negative prompt này được tăng cường để hiệu quả hơn trong việc loại bỏ các vấn đề phổ biến.
*   **Tham số tạo ảnh**: Các tham số như `num_inference_steps` (số bước suy luận, ví dụ: 20, 30, 50) và `guidance_scale` (mức độ tuân thủ prompt, ví dụ: 12.0) có thể được điều chỉnh để kiểm soát chất lượng và sự đa dạng của ảnh tạo ra. Script thử nghiệm tạo ảnh với các số bước suy luận khác nhau.
*   Ảnh được tạo ra và lưu vào thư mục đầu ra (`/kaggle/working/generated/`). Các phiên bản ảnh với số bước suy luận khác nhau được hiển thị để so sánh.
