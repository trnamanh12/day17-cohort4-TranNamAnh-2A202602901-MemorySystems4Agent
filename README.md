# Phase 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

```
.
├── README.md        # giới thiệu track (file này)
├── Guide.md         # hướng dẫn từng bước
├── Rubric.md        # tiêu chí chấm điểm
├── data/            # dữ liệu benchmark dùng chung
│   ├── conversations.json
│   └── advanced_long_context.json
└── src/             # bản scaffold dành cho sinh viên (pseudocode + TODO)
    ├── model_provider.py
    ├── config.py
    ├── memory_store.py
    ├── agent_baseline.py
    ├── agent_advanced.py
    ├── benchmark.py
    └── test_agents.py
```

Khi chạy, agent sẽ ghi trạng thái (ví dụ `state/profiles/<user>/User.md`) vào thư mục `state/`. Thư mục này đã nằm trong `.gitignore`.

### Vai trò từng file trong `src/`

Các file được liệt kê theo thứ tự nên triển khai:

| File | Vai trò | Thành phần chính |
|---|---|---|
| `model_provider.py` | Khởi tạo chat model cho từng provider | `ProviderConfig`, `normalize_provider()`, `build_chat_model()` |
| `config.py` | Cấu hình chung của lab | `LabConfig` (đường dẫn, ngưỡng compact, model chính + judge), `load_config()` |
| `memory_store.py` | Lõi memory layer | `estimate_tokens()`, `UserProfileStore` (read/write/edit `User.md`), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `agent_baseline.py` | Agent A: chỉ nhớ trong cùng thread | `BaselineAgent.reply()`, `token_usage()`, `prompt_token_usage()` |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact | `AdvancedAgent.reply()`, `_reply_offline()`, `_estimate_prompt_context_tokens()`, `_offline_response()` |
| `benchmark.py` | So sánh hai agent trên hai bộ dữ liệu | `run_agent_benchmark()`, `recall_points()`, `heuristic_quality()`, `format_rows()` |
| `test_agents.py` | Kiểm chứng hành vi memory | test `User.md`, compact trigger, cross-session recall, giảm prompt load |

### Luồng xử lý một lượt của Advanced Agent

```
message người dùng
  → extract_profile_updates()      # trích fact ổn định: tên, nơi ở, nghề, style...
  → ghi vào User.md                # persistent memory
  → CompactMemoryManager.append()  # short-term memory, tự compact khi vượt ngưỡng
  → prompt = User.md + summary + recent messages
  → sinh câu trả lời → cập nhật bộ đếm token
```

Baseline Agent chỉ giữ danh sách message theo `thread_id`. Sang thread mới, nó **phải quên** toàn bộ fact cũ.

Cả hai agent có **chế độ offline** cho kết quả lặp lại được, để benchmark và test chạy không cần API key. Khi cấu hình credentials và cài integration tương ứng, chế độ live gọi chat model qua LangChain; ứng dụng vẫn quản lý lịch sử thread, profile và compaction.

## Dữ liệu benchmark

| File | Nội dung | Mục tiêu |
|---|---|---|
| `data/conversations.json` | 10 hội thoại khoảng 10 lượt, user `dungct`, kèm `recall_questions` | Standard benchmark: đo recall qua nhiều phiên bình thường |
| `data/advanced_long_context.json` | 1 hội thoại 16 lượt rất dài, user `dungct_stress` | Long-context stress benchmark: ép compact xảy ra nhiều lần |

Mỗi hội thoại có dạng:

```json
{
  "id": "conv-01",
  "user_id": "dungct",
  "turns": ["...", "..."],
  "recall_questions": [
    { "question": "...", "expected_contains": ["DũngCT", "cà phê sữa đá"] }
  ]
}
```

`recall_questions` được hỏi ở **thread mới**. Điểm recall dựa trên số chuỗi trong `expected_contains` xuất hiện trong câu trả lời.

Dữ liệu cố tình chứa các tình huống khó:

- **correction**: nơi ở đổi giữa Đà Nẵng và Huế, agent phải giữ fact mới nhất
- **nhiễu**: "Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là câu đùa
- **ngữ cảnh dài**: nhiều đoạn tin tức dài trong stress test để làm lộ chi phí prompt của baseline

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Nếu muốn chạy chế độ live với LLM thật, hãy tạo file `.env` ở root repo (đã nằm trong `.gitignore`). Tên biến môi trường do các bạn quyết định khi viết `load_config()`. Ví dụ:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

## Chạy benchmark và test

Sau khi hoàn thiện `src/`, chạy từ root repo:

```bash
python src/benchmark.py
```

```bash
pytest src/test_agents.py -v
```

Benchmark cần in ra hai bảng: **Standard Benchmark** và **Long-Context Stress Benchmark**. Mỗi bảng so sánh Baseline với Advanced theo đủ 6 cột trong phần "Chỉ số benchmark cần hiểu".

## Kết quả và phân tích

Dưới đây là bảng số liệu thu thập được khi chạy benchmark ở chế độ offline trên hai tập dữ liệu mẫu:

| Bộ benchmark | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 1,664 | 16,923 | 0% | 30% | 0 | 0 |
| Standard | Advanced | 3,104 | 27,532 | 100% | 100% | 331 | 0 |
| Long-context stress | Baseline | 406 | 23,048 | 0% | 30% | 0 | 0 |
| Long-context stress | Advanced | 539 | 13,511 | 100% | 100% | 239 | 3 |

*(Ghi chú: `Response quality` được tính bằng heuristic offline: 70% dựa trên việc xuất hiện các fact mong muốn và 30% dựa trên độ súc tích của câu trả lời. Số lượng token được ước lượng theo quy tắc ký tự chia bốn để tiện so sánh tương đối giữa hai agent).*

---

### 1. Phân tích trade-off: Khi nào Advanced thắng, khi nào Baseline tối ưu hơn?

Khi nhìn vào bảng kết quả, có hai câu chuyện đối lập rất rõ ràng giữa hội thoại ngắn và hội thoại dài:

- **Ở hội thoại thông thường (Standard Benchmark):**
  - **Khả năng nhớ dài hạn (Recall):** Baseline hoàn toàn "mất trí nhớ" (0% recall) khi bước sang một thread mới vì nó chỉ lưu ngữ cảnh cục bộ trong từng session. Ngược lại, Advanced Agent đạt recall tuyệt đối (100%) nhờ việc duy trì file hồ sơ bền vững `User.md`.
  - **Chi phí Prompt Tokens:** Rõ ràng "không có bữa trưa nào miễn phí". Để có được trí nhớ dài hạn, Advanced Agent tốn nhiều prompt tokens hơn Baseline khá nhiều (27,532 so với 16,923 tokens). Nguyên nhân là ở mỗi lượt chat, Advanced đều phải đính kèm toàn bộ nội dung của `User.md` vào prompt đầu vào, tạo thành một khoản chi phí cố định (overhead) lặp đi lặp lại. Trong khi đó, các cuộc trò chuyện ở tập standard còn quá ngắn nên cơ chế compact chưa kịp kích hoạt lần nào.

- **Ở hội thoại kéo dài (Long-Context Stress Benchmark):**
  - Câu chuyện lập tức đảo chiều khi cuộc trò chuyện trở nên rất dài (16 turns với nhiều đoạn văn bản lớn). Baseline Agent bắt đầu bộc lộ nhược điểm chí mạng: nó phải kéo theo toàn bộ lịch sử từ đầu đến cuối, khiến lượng prompt token xử lý tăng vọt lên hơn 23,000 tokens.
  - Ngược lại, Advanced Agent đã tự động kích hoạt nén bộ nhớ (compaction) **3 lần**. Bằng cách tóm tắt các lượt trao đổi cũ và chỉ giữ lại cửa sổ các tin nhắn gần nhất, Advanced đã cắt giảm lượng prompt tokens phải xử lý xuống còn 13,511 tokens — **tiết kiệm khoảng 41% chi phí ngữ cảnh** so với Baseline.
  - **Điểm mấu chốt cần lưu ý:** Compact Memory chủ yếu tối ưu cho **Prompt Tokens Processed** (ngữ cảnh gửi vào model), chứ không đảm bảo giảm **Agent Tokens Only** (token câu trả lời sinh ra). Thậm chí agent tokens của Advanced còn nhỉnh hơn một chút vì câu trả lời của nó có kèm các fact nhớ lại từ hồ sơ.

---

### 2. Sự tăng trưởng của bộ nhớ và các rủi ro đi kèm

- Kích thước file `User.md` tăng khá khiêm tốn: khoảng 331 bytes ở bộ Standard và 239 bytes ở bộ Stress test. Định dạng Markdown phân tầng giúp thông tin vừa gọn nhẹ vừa dễ đọc đối với cả con người lẫn LLM.
- **Những rủi ro thực tế cần đối mặt:**
  1. *Nguy cơ phình to bộ nhớ (Memory Bloat):* Nếu người dùng tương tác qua nhiều tháng hoặc nhiều năm, file `User.md` sẽ dần trở nên quá nặng, vô tình biến khoản overhead mỗi lượt chat thành gánh nặng token khổng lồ.
  2. *Ảo giác và lưu nhầm thông tin (False Positives):* Bộ bóc tách thông tin nếu chỉ dựa vào từ khóa đơn giản rất dễ hiểu nhầm câu hỏi, câu đùa hoặc các chuyến đi tạm thời thành sự thật vĩnh viễn.

---

### 3. Kinh nghiệm thực tế khi phát triển các cơ chế bảo vệ (Bonus 90 - 100 điểm)

Để giải quyết các rủi ro trên và đưa hệ thống lên mức hoàn thiện tiệm cận production, mình đã xây dựng thêm 3 cơ chế mở rộng:

#### A. Lọc câu hỏi và câu giả định bằng Confidence Threshold
- **Vấn đề thực tế:** Khi test với các câu nói tự nhiên, người dùng rất hay hỏi xác nhận (*"Mình đang ở Hà Nội phải không?"*) hoặc đưa ra giả định (*"Nếu mình làm product manager thì sao?"*). Một bộ trích xuất thông thường sẽ bắt nhầm chữ "Hà Nội" hay "product manager" và ghi ngay vào hồ sơ.
- **Cách giải quyết:** Hàm `extract_structured_facts()` sẽ quét ngữ cảnh xung quanh để tính điểm tin cậy `confidence` (từ 0.0 đến 1.0). Những câu có dấu hỏi `?`, từ nghi vấn (*phải không, đúng không, ở đâu...*) hoặc từ giả định (*nếu, giả sử, ước gì...*) sẽ bị phạt điểm xuống dưới 0.4. Chỉ những câu khẳng định rõ ràng đạt `confidence >= 0.7` mới được phép ghi vào `User.md`.
- **Trade-off:** Cách này giữ cho hồ sơ sạch sẽ và không tốn thêm token, nhưng có thể bỏ sót một vài câu khẳng định nếu người dùng viết câu quá dài dòng hoặc dùng từ ngữ nước đôi.

#### B. Xử lý xung đột và đính chính (Conflict Handling & Audit)
- **Vấn đề thực tế:** Người dùng thường xuyên thay đổi thông tin (ví dụ: ban đầu ở Đà Nẵng, sau đó đính chính chuyển về Huế). Nếu cứ lưu dồn dập, hồ sơ sẽ chứa hai thông tin mâu thuẫn nhau.
- **Cách giải quyết:** Với các trường đơn trị (`current_location`, `profession`...), hàm `upsert_fact()` sẽ ghi đè giá trị mới lên giá trị cũ để context luôn nhất quán, đồng thời đẩy giá trị cũ vào lịch sử kiểm toán `superseded_facts()`. Nhờ đó, agent luôn nhớ đúng fact mới nhất (đạt 100% recall) mà vẫn có vết audit khi cần tra cứu lại.
- **Trade-off:** Cần giữ dữ liệu audit tách biệt khỏi nội dung Markdown gửi cho LLM để không làm tăng prompt context không cần thiết.

#### C. Chống phình bộ nhớ bằng Memory Decay & Pruning
- **Vấn đề thực tế:** Để giải quyết bài toán `User.md` phình to theo thời gian, hệ thống cần biết thông tin nào quan trọng và thông tin nào có thể bỏ đi.
- **Cách giải quyết:** Bổ sung phương thức `prune_stale_facts(max_facts)`: ưu tiên bảo vệ các thông tin cốt lõi (*tên, nơi ở, nghề nghiệp, phong cách trả lời, sở thích*) và tự động cắt tỉa các ghi chú phụ khi tổng số fact vượt quá ngưỡng quy định. Đồng thời có phương thức `decay_facts()` để chủ động loại bỏ những thông tin đã cũ hoặc không còn được nhắc lại.
- **Trade-off:** Việc dọn bớt thông tin giúp khống chế chi phí token lâu dài, nhưng đánh đổi lại là hệ thống có thể quên đi một số sở thích ngách nếu người dùng không nhắc lại chúng thường xuyên.

## Cách dùng repo này

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.
