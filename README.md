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
  - Trạng thái ban đầu: [[2, 6, 5], [0, 8, 7], [4, 3, 1]].
  - Trạng thái mục tiêu: [[1, 2, 3], [4, 5, 6], [7, 8, 0]].
  - Hành động: Di chuyển ô trống lên, xuống, trái, hoặc phải.
  - Hàm chi phí: Chi phí của mỗi hành động - 1 cho mỗi di chuyển trong 8-Puzzle.
- Solution: Là danh sách các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu.

**Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:**

#### BFS:

![bfs](https://github.com/user-attachments/assets/1684b10b-95a1-4f45-831b-649b4ef6d163)

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
  - Trạng thái ban đầu: [[2, 6, 5], [0, 8, 7], [4, 3, 1]].
  - Trạng thái mục tiêu: [[1, 2, 3], [4, 5, 6], [7, 8, 0]].
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
  - Trạng thái ban đầu: [[2, 6, 5], [0, 8, 7], [4, 3, 1]].
  - Trạng thái mục tiêu: [[1, 2, 3], [4, 5, 6], [7, 8, 0]].
  - Hành động: Di chuyển ô trống lên, xuống, trái, hoặc phải.
  - Hàm chi phí: Chi phí của mỗi hành động - 1 cho mỗi di chuyển trong 8-Puzzle.
- Solution: Là danh sách các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu.

**Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:**

#### Beam Search:

![beam](https://github.com/user-attachments/assets/966485db-6413-418f-8e1d-3a86934414cb)

**Nhận xét về hiệu suất:**

- Simple Hill Climbing: Hiệu suất kém trong 8-Puzzle do thiếu cơ chế thoát khỏi đỉnh cục bộ, không phù hợp với bài toán cần đường đi chính xác.
- Steepest-Ascent Hill Climbing: Tốt hơn Simple Hill Climbing, nhưng vẫn không hiệu quả do không gian tìm kiếm phức tạp của 8-Puzzle.
- Stochastic Hill Climbing: Tính ngẫu nhiên không đủ mạnh để giải bài toán 8-Puzzle, hiệu suất kém.
- Simulated Annealing: Có tiềm năng hơn Hill Climbing, nhưng thất bại do tham số không tối ưu.
- Beam Search: thuật toán duy nhất trong nhóm thành công, với thời gian cực nhanh (0.02 giây) nhờ giới hạn khám phá trong k trạng thái. Tuy nhiên, đường đi 29 bước chưa tối ưu, phù hợp khi ưu tiên tốc độ hơn.
- Genetic Algorithm: Không phù hợp với 8-Puzzle do bản chất bài toán cần chuỗi di chuyển tuần tự, không phải tối ưu hóa cấu hình.
  Tóm lại: Không áp dụng được do không tìm thấy giải pháp. Beam Search thành công nhờ giữ k trạng thái, khám phá rộng hơn Hill Climbing. Tuy nhiên không tối ưu, phụ thuộc k. Nếu k quá nhỏ, có thể thất bại như các thuật toán khác.

### 2.4. Complex Environment (AND-OR Search, Belief State Search, Searching with Partial Observation)

**Các thành phần chính của bài toán tìm kiếm và solution**

  - Không gian trạng thái: Tập hợp tất cả các cấu hình có thể của bảng 8-Puzzle. Trong môi trường phức tạp, bao gồm belief states (tập hợp trạng thái khả thi).
  - Trạng thái ban đầu: [[2, 6, 5], [0, 8, 7], [4, 3, 1]]. Cấu hình ban đầu hoặc belief state ban đầu.
  - Trạng thái mục tiêu: [[1, 2, 3], [4, 5, 6], [7, 8, 0]]. Cấu hình mong muốn hoặc belief state chứa mục tiêu.
  - Hành động: Di chuyển ô trống lên, xuống, trái, hoặc phải.
- Solution: Là danh sách các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu.

**Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi:**

#### Belief State:

![bs](https://github.com/user-attachments/assets/9a503027-eee3-4727-b530-ec08d18b3e78)

#### Searching with Partial Observation:

![swpo](https://github.com/user-attachments/assets/07dcd7da-7902-4eea-981d-0689cd582a5c)

**So sánh hiệu suất các thuật toán:**

![image](https://github.com/user-attachments/assets/f45cdc51-a3b9-420c-9fcf-cff98147c20d)

**Nhận xét về hiệu suất:**

- AND-OR Search: Kkhông tìm được giải pháp do không phù hợp với 8-Puzzle chuẩn.
- Belief State Search: Tốn thời gian do quản lý belief states, nhưng thất bại trong việc đạt đích dù có 23 bước (bằng đường đi tối ưu). Điều này cho thấy thuật toán có tiềm năng nhưng triển khai chưa đúng.
- Searching with Partial Observation: Đạt được đường đi tối ưu (23 bước) với thời gian hợp lý (1.36 giây).
  Tóm lại: Searching with Partial Observation hiệu quả, nhưng không cần thiết cho 8-Puzzle chuẩn. Belief State Search có thể tận dụng tiềm năng. AND-OR Search không phù hợp.

### 2.5. CSPS (AC-3, Backtracking, Backtracking with Forward Checking)

**Các thành phần chính của bài toán tìm kiếm và solution**

  - Variables: mỗi biến đại diện cho một giá trị từ 0 đến 8 cần được gán vào một vị trí trên bảng 3x3.
  - Domains (Miền giá trị) Mỗi biến (số từ 0 đến 8) có miền là các vị trí từ 0 đến 8 (trên bảng 3x3). Tức là mỗi số có thể nằm ở bất kỳ ô nào trên bàn cờ.
  - Constraints (Ràng buộc) Điều kiện là mỗi số chỉ được gán vào đúng một vị trí duy nhất, không được trùng nhau (Trong ac_3, ràng buộc được lọc qua hàm revise, đảm bảo không có hai biến nào có thể trùng vị trí hợp) lệ.
- Solution: Không có 2 biến nào trùng vị trí (ràng buộc all-different), dựng được bàn cờ đúng bằng trạng thái goal

**Nhận xét về hiệu suất:**

- AC-3: Không phù hợp với 8-Puzzle vì không giải quyết được bản chất tìm kiếm đường đi.
- Backtracking Search: Không phù hợp với 8-Puzzle vì không giải quyết được bản chất tìm kiếm đường đi.
- Backtracking with Forward Checking: Dù nhanh hơn Backtracking nhưng vẫn không phù hợp.

### 2.6. Reinforcement Learning (Q-Learning)

**Các thành phần chính của bài toán tìm kiếm và solution**

  - Không gian trạng thái: Tập hợp tất cả các cấu hình có thể của bảng 8-Puzzle.
  - Trạng thái ban đầu: [[2, 6, 5], [0, 8, 7], [4, 3, 1]].
  - Trạng thái mục tiêu: [[1, 2, 3], [4, 5, 6], [7, 8, 0]].
  - Hành động: Di chuyển ô trống lên, xuống, trái, hoặc phải.
  - Reward:	Số điểm nhận được khi chuyển trạng thái, -1 (bình thường), +200 (nếu đạt goal)
  - Q-function	Hàm đánh giá giá trị hành động tại một trạng thái cụ thể
  - Learning Algorithm: Q-value để cải thiện chính sách qua từng bước
- Solution: Tìm một chuỗi hành động (path) dẫn từ trạng thái start đến goal_state sao cho tối đa hóa tổng phần thưởng nhận được.

**Nhận xét về hiệu suất:**

- Q-Learning: Không phù hợp với 8-Puzzle do yêu cầu huấn luyện lâu và không gian trạng thái lớn.

## Kết luận

Dự án 8-Puzzle Solver đã thành công trong việc triển khai 6 nhóm thuật toán với giao diện đồ họa trực quan. Kết quả đạt được bao gồm:

- **GUI**: Người dùng dễ dàng quan sát tiến trình giải qua animation.
- **Triển khai thuật toán**: Đa dạng từ cơ bản (BFS, DFS) đến nâng cao (Belief State Search, Q-Learning).
- **Hiệu suất**:
  - **Nhanh nhất**: Greedy Search (Informed Search) và Beam Search (Local Search) - 0.02 giây (cả hai). Greedy nhanh hơn một chút do chỉ chọn một trạng thái lân cận, nhưng Beam Search cho đường đi ngắn hơn (29 bước so với 49 bước).
  - **Chậm nhất**: IDS (Uninformed Search). Do lặp lại tìm kiếm với độ sâu tăng dần, dẫn đến khám phá lặp lại nhiều nút, đặc biệt với độ sâu tối ưu 23 bước.
- **Kết luận**:
  - A* (0.2 giây, 23 bước) là lựa chọn tối ưu, cân bằng tốc độ, tính hoàn chỉnh, và tính tối ưu.
  - Greedy, Beam Search (0.02 giây) phù hợp khi ưu tiên tốc độ, nhưng không tối ưu.
  - Chậm và không thực tế: IDS (5.94 giây, 29 bước), DFS (7113 bước).
  - Không phù hợp: CSP, Q-Learning, AND-OR, Hill Climbing, Genetic Algorithm thất bại do mô hình hoặc triển khai không phù hợp với 8-Puzzle (fully observable, deterministic).
  - Belief State Search, Q-Learning có thể thành công nếu khắc phục lỗi triển khai hoặc tăng huấn luyện.

**Hướng phát triển**: Tối ưu hóa, cải thiện các thuật toán.

## Link video báo cáo
https://drive.google.com/drive/folders/1hSSgNApQ3fzj2GGYJvvBUyq5nF_NH10B?usp=sharing

## Link github
https://github.com/Quan-Man/Personal-Project-AI.git
