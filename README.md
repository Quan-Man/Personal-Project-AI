# Personal-Project-AI: 8-Puzzle Solver with algorithms
## Họ và tên: Quan Gia Mẫn  -  MSSV: 23133042
## Lớp ARIN330585_05 xin chuyển qua lớp chiều thứ 5 (HKII 2024-2025)

## 1. Mục tiêu

- Xây dựng giao diện đồ họa để hiển thị trạng thái ban đầu, trạng thái mục tiêu, và các bước giải chi tiết.

- Triển khai 6 nhóm thuật toán:

  - Uninformed Search: BFS, DFS, UCS, IDS.
  - Informed Search: A*, IDA*, Greedy.
  - Local Search: Simple Hill Climbing, Steepest-Ascent Hill Climbing, Stochastic Hill Climbing, Simulated Annealing, Beam Search, Genetic Algorithm.
  - Complex Environment: AND-OR Search, Belief State Search, Searching with Partial Observation.
  - CSPS: AC-3, Backtracking, Backtracking with Forward Checking.
  - Reinforcement Learning: Q-Learning.

- Đánh giá hiệu suất của các thuật toán dựa trên số bước và thời gian thực hiện.

### Dữ liệu thử nghiệm

Các thuật toán được thử nghiệm với trạng thái ban đầu và mục tiêu sau:

- **Trạng thái ban đầu**: `[[2, 6, 5], [0, 8, 7], [4, 3, 1]]`

  ![image](https://github.com/user-attachments/assets/c8fb5961-f4f2-418c-9563-1bfd0a018ed3)

- **Trạng thái mục tiêu**: `[[1, 2, 3], [4, 5, 6], [7, 8, 0]]`

  ![image](https://github.com/user-attachments/assets/98413488-d190-4264-9d41-5a7f83353185)

## 2. Nội dung

### 2.1. Uninformed Search (BFS, DFS, UCS, IDS)

**Các thành phần chính của bài toán tìm kiếm và solution**

  - Không gian trạng thái: Tập hợp tất cả các cấu hình có thể của bảng 8-Puzzle (3x3 grid với 8 ô số từ 1-8 và 1 ô trống). Mỗi trạng thái là một cách sắp xếp các ô.
  - Trạng thái ban đầu: Lưới 3x3 với 8 ô số và 1 ô trống (0).
  - Trạng thái mục tiêu: Lưới 3x3 với các ô số từ 1-8 và ô trống ở vị trí cuối.
  - Hành động: Di chuyển ô trống lên, xuống, trái, hoặc phải.
  - Hàm chi phí: Chi phí của mỗi hành động - 1 cho mỗi di chuyển trong 8-Puzzle.
- Solution: Là danh sách các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu.

**Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:**

#### BFS:

https://github.com/user-attachments/assets/ce0b6e04-d26b-4673-b69b-e7034b8ec487

#### DFS:

![image](https://github.com/user-attachments/assets/dcca876f-1e60-4aab-94b0-b8d67451fadc)

#### UCS:

![ucs](https://github.com/user-attachments/assets/d4058bb5-9cab-4d15-a51b-dbc74cc43175)

#### IDS:

![ids](https://github.com/user-attachments/assets/bceaad04-d580-4db2-8c6c-5c9d0c48a556)

**So sánh hiệu suất các thuật toán:**

![image](https://github.com/user-attachments/assets/6536b101-63c7-4418-863c-37a039058ca9)

**Nhận xét về hiệu suất:**

- BFS: Đáng tin cậy, đảm bảo đường đi ngắn nhất, nhưng thời gian chạy khá cao do độ phức tạp không gian và thời gian tăng theo cấp số nhân.
- DFS: Không phù hợp cho 8-Puzzle do thiếu tính tối ưu và dễ đi vào nhánh sâu không hiệu quả. Thời gian nhanh là ưu điểm duy nhất, nhưng không bù đắp được chất lượng giải pháp.
- UCS: Đáng tin cậy như BFS nhưng kém hiệu quả hơn về thời gian do chi phí tính toán cao hơn.
- IDS: Tiết kiệm bộ nhớ hơn BFS/UCS, nhưng thời gian chạy cao và kết quả không tối ưu.

  Tóm lại: BFS và UCS hoạt động đúng như kỳ vọng: tìm đường đi ngắn nhất, nhưng tốn thời gian và bộ nhớ. DFS nhanh nhưng không thực tế do đường đi quá dài. IDS có vấn đề (29 bước, 5.94 giây), có thể do lỗi triển khai hoặc không tối ưu hóa.

### 2.2. Informed Search (A*, IDA*, Greedy)

**Các thành phần chính của bài toán tìm kiếm và solution**

- Không gian trạng thái: Tập hợp tất cả các cấu hình có thể của bảng 8-Puzzle (3x3 grid với 8 ô số từ 1-8 và 1 ô trống). Mỗi trạng thái là một cách sắp xếp các ô.
  - Trạng thái ban đầu: Lưới 3x3 với 8 ô số và 1 ô trống (0).
  - Trạng thái mục tiêu: Lưới 3x3 với các ô số từ 1-8 và ô trống ở vị trí cuối.
  - Hành động: Di chuyển ô trống lên, xuống, trái, hoặc phải.
  - Hàm chi phí: Chi phí của mỗi hành động - 1 cho mỗi di chuyển trong 8-Puzzle.
- Solution: Là danh sách các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu.

**Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:**

#### A*:

![astar](https://github.com/user-attachments/assets/df78be3a-3c5c-485c-b65d-42abed985fa4)

#### IDA*:

![ida](https://github.com/user-attachments/assets/0ce3630d-3b4e-4e3b-aa7b-7cdda0534da2)

#### Greedy:

![greedy](https://github.com/user-attachments/assets/fc59ef53-35e6-4e67-bb83-9a006e055e7e)

**So sánh hiệu suất các thuật toán:**

![image](https://github.com/user-attachments/assets/cf557a89-4d7b-41a3-b370-1015792a581f)

**Nhận xét về hiệu suất:**

- A*: Là lựa chọn xuất sắc, cân bằng giữa tính tối ưu và tốc độ, vượt trội so với BFS/UCS nhờ heuristic.
- IDA*: Đáng tin cậy về tính tối ưu, nhưng thời gian chạy cao làm nó kém hấp dẫn hơn A* trong trường hợp này.
- Greedy: Phù hợp khi cần tốc độ cao và chấp nhận giải pháp không tối ưu, nhưng không nên dùng khi cần đường đi ngắn nhất.

  Tóm lại: Hiệu suất thực tế của A* phù hợp và tốt hơn lý thuyết, đặc biệt về thời gian, nhờ heuristic Manhattan và kiểm tra visited hiệu quả. IDA* đạt tính tối ưu như lý thuyết, nhưng thời gian thực tế (2.34 giây) cao hơn kỳ vọng. Greedy nhanh, không hoàn chỉnh, không tối ưu.

### 2.3. Local Search (Steepest-Ascent Hill Climbing, Stochastic Hill Climbing, Simple Hill Climbing, Beam Search, Simulated Annealing, Genetic Algorithm)

**Các thành phần chính của bài toán tìm kiếm và solution**

  - Không gian trạng thái: Tập hợp tất cả các cấu hình có thể của bảng 8-Puzzle (3x3 grid với 8 ô số từ 1-8 và 1 ô trống). Mỗi trạng thái là một cách sắp xếp các ô.
  - Trạng thái ban đầu: Lưới 3x3 với 8 ô số và 1 ô trống (0).
  - Trạng thái mục tiêu: Lưới 3x3 với các ô số từ 1-8 và ô trống ở vị trí cuối.
  - Hành động: Di chuyển ô trống lên, xuống, trái, hoặc phải.
  - Hàm chi phí: Chi phí của mỗi hành động - 1 cho mỗi di chuyển trong 8-Puzzle.
- Solution: Là danh sách các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu.

**Nhận xét về hiệu suất:**

- Beam Search: Hiệu quả với beam_width nhỏ, giảm bộ nhớ so với tìm kiếm toàn cục, nhưng có thể bỏ lỡ lời giải nếu beam_width quá hẹp hoặc không gian trạng thái quá lớn.
- Simple Hill Climbing: Nhanh và đơn giản, nhưng dễ bị kẹt ở cực trị cục bộ, dẫn đến thất bại nếu không có đường đi trực tiếp giảm heuristic.
- Steepest-Ascent Hill Climbing: Cải tiến hơn Simple Hill Climbing bằng cách chọn trạng thái tốt nhất trong số các lân cận, nhưng vẫn dễ bị kẹt ở cực trị cục bộ và yêu cầu nhiều phép tính hơn.
- Stochastic Hill Climbing: Linh hoạt hơn nhờ chọn ngẫu nhiên trong số các trạng thái tốt hơn, giảm nguy cơ kẹt ở cực trị cục bộ, nhưng hiệu suất phụ thuộc vào may rủi.
- Genetic Algorithm: Rất linh hoạt và có thể tìm ra lời giải trong không gian phức tạp, nhưng cần nhiều thời gian huấn luyện và tài nguyên do sử dụng quần thể và tiến hóa qua nhiều thế hệ.
- Simulated Annealing: Khắc phục nhược điểm của Hill Climbing bằng cách cho phép chấp nhận trạng thái xấu hơn với xác suất, tránh được cực trị cục bộ. Hiệu suất tốt hơn Hill Climbing trong các trạng thái phức tạp, nhưng phụ thuộc vào tham số nhiệt độ (T, cooling_rate) và có thể chậm nếu cần nhiều bước để hội tụ.

### 2.4. Complex Environment (AND-OR Search, Belief State Search, Searching with Partial Observation)

**Các thành phần chính của bài toán tìm kiếm và solution**

  - Không gian trạng thái: Tập hợp tất cả các cấu hình có thể của bảng 8-Puzzle Trong môi trường phức tạp, bao gồm belief states (tập hợp trạng thái khả thi).
  - Trạng thái ban đầu: Lưới 3x3 với 8 ô số và 1 ô trống (0). Cấu hình ban đầu hoặc belief state ban đầu.
  - Trạng thái mục tiêu: Lưới 3x3 với các ô số từ 1-8 và ô trống ở vị trí cuối. Cấu hình mong muốn hoặc belief state chứa mục tiêu.
  - Hành động: Di chuyển ô trống lên, xuống, trái, hoặc phải.
- Solution: Là danh sách các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu.

**Nhận xét về hiệu suất:**

- AND-OR Search: Phù hợp với các bài toán có phụ thuộc logic phức tạp, nhưng hiệu suất thấp trong 8-puzzle do không tận dụng được cấu trúc tuyến tính, và dễ bị giới hạn bởi độ sâu tối đa.
- Belief State Search: Hiệu quả trong môi trường không chắc chắn (partial observation), nhưng tốn nhiều bộ nhớ và thời gian để quản lý và cập nhật belief states, đặc biệt với số lượng ô không xác định lớn.
- Searching with Partial Observation: Linh hoạt và thích nghi tốt với thông tin quan sát từng bước, nhưng hiệu suất phụ thuộc vào độ chính xác của quan sát và có thể thất bại nếu belief states không còn khả thi.

### 2.5. CSPS (AC-3, Backtracking, Backtracking with Forward Checking)

**Các thành phần chính của bài toán tìm kiếm và solution**

- Thành phần chính:

  - Biến: Các ô trên lưới 3x3.
  - Miền giá trị: Các số từ 0-8.
  - Ràng buộc: Mỗi số phải duy nhất (uniqueness constraint).

- Solution: Đường đi từ trạng thái ban đầu đến mục tiêu, thỏa mãn tất cả ràng buộc.

  - Không gian trạng thái: Tập hợp tất cả các cấu hình có thể của bảng 8-Puzzle (3x3 grid với 8 ô số từ 1-8 và 1 ô trống). Mỗi trạng thái là một cách sắp xếp các ô.
  - Trạng thái ban đầu: Lưới 3x3 với 8 ô số và 1 ô trống (0).
  - Trạng thái mục tiêu: Lưới 3x3 với các ô số từ 1-8 và ô trống ở vị trí cuối.
  - Hành động: Di chuyển ô trống lên, xuống, trái, hoặc phải.
  - Hàm chi phí: Chi phí của mỗi hành động - 1 cho mỗi di chuyển trong 8-Puzzle.
- Solution: Là danh sách các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu.

**Nhận xét về hiệu suất:**

- AC-3: Hiệu quả hơn nhờ duy trì tính nhất quán cung trong suốt quá trình tìm kiếm, giảm số trạng thái cần kiểm tra. Tuy nhiên, việc chạy có thể tốn thời gian với các trạng thái phức tạp.
- Backtracking Search: Cơ bản và dễ triển khai, nhưng hiệu suất thấp do phải thử nghiệm nhiều trạng thái không hợp lệ, đặc biệt với không gian trạng thái lớn.
- Backtracking with Forward Checking: Cải thiện hiệu suất bằng cách loại bỏ sớm các giá trị không hợp lệ, giảm số lượng trạng thái cần kiểm tra, nhưng vẫn có thể chậm nếu ràng buộc phức tạp.

### 2.6. Reinforcement Learning (Q-Learning)

**Các thành phần chính của bài toán tìm kiếm và solution**

- Thành phần chính: Sử dụng Q-Table, phần thưởng, và chiến lược epsilon-greedy để học.
- Solution: Đường đi học được từ huấn luyện.

**Nhận xét về hiệu suất:**

- Q-Learning: Hiệu quả khi được huấn luyện tốt với số lượng episode lớn, có khả năng học đường đi tối ưu theo thời gian. Tuy nhiên, cần nhiều thời gian huấn luyện ban đầu và hiệu suất ban đầu có thể kém do khám phá ngẫu nhiên (epsilon-greedy).

## Kết luận

Dự án 8-Puzzle Solver đã thành công trong việc triển khai 6 nhóm thuật toán với giao diện đồ họa trực quan. Kết quả đạt được bao gồm:

- **GUI**: Người dùng có thể nhập trạng thái tùy chỉnh, quan sát tiến trình giải qua animation.
- **Triển khai thuật toán**: Đa dạng từ cơ bản (BFS, DFS) đến nâng cao (Belief State Search, Q-Learning).
- **Hiệu suất**:
  - **Nhanh nhất**: A* và IDA* nhờ heuristic hiệu quả.
  - **Chậm nhất**: GA và Q-Learning do cần huấn luyện.
  - Belief State Search hiệu quả trong môi trường không chắc chắn, nhưng tốn tài nguyên.
- **Kết luận**:
  - Các thuật toán heuristic-based (A*, IDA*) phù hợp nhất cho 8-puzzle nhờ cân bằng giữa độ tối ưu và hiệu suất.
  - Các thuật toán như Genetic Algorithm và Q-Learning phù hợp với môi trường phức tạp, nhưng cần tối ưu hóa thêm.
  - CSPs (như MAC) hiệu quả khi không gian trạng thái nhỏ, nhưng không tối ưu cho 8-puzzle.

**Hướng phát triển**: Tối ưu hóa thuật toán (ví dụ: giảm bộ nhớ cho A\*).

## Link video báo cáo
https://drive.google.com/drive/folders/1hSSgNApQ3fzj2GGYJvvBUyq5nF_NH10B?usp=sharing

## Link github
https://github.com/Quan-Man/Personal-Project-AI.git
