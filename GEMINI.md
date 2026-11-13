# GEMINI.md

See [AGENTS.md](file:///Users/harsh/Code/auv/coralseg/AGENTS.md) for full project rules and guidelines.

### Mandatory Rules:

#### 1. Push to GitHub on Completion of Every Step
On completion of every development step, task, bug fix, or milestone:
1. Verify implementation (**Implement → Test → Fix → Re-test**).
2. Check `git status` to ensure only intended changes are staged.
3. Commit with a clear, descriptive message (`git commit -m "..."`).
4. Push immediately to GitHub (`git push origin main` or active branch).

#### 2. Do NOT Run Full Heavy Pipeline Locally for Testing
- Never run the complete heavy pipeline on the user's local PC for testing every time.
- **On every MAJOR change**:
  1. First try to run/test on **Google Colab** (using Colab CLI GPU session, which is default).
  2. Or create and test a **smaller / scaled-down version** (minimal frames, dummy data, or lightweight mock) that can safely run locally on the user's PC without exhausting resources.
