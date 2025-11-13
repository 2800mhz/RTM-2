from typing import List, Optional
import yaml
import os
import torch
import torch.distributed as dist
import pydantic
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader

class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]
    num_samples: Optional[int] = None
    visualize: bool = False
    visualize_count: int = 5
    show_only_errors: bool = False  # Sadece yanlış çözümleri göster
    detailed_stats: bool = True  # Detaylı istatistikler
    save_html_report: bool = False  # HTML rapor oluştur

def visualize_sudoku(puzzle, solution, prediction, idx, is_correct, difficulty_score=None):
    """Sudoku'yu terminalde görselleştir"""
    status = "✅ TAMAMEN DOĞRU" if is_correct else "❌ YANLIŞ"
    print(f"\n{'='*60}")
    print(f"🎯 Sudoku #{idx + 1} - {status}")
    if difficulty_score is not None:
        print(f"🔥 Zorluk Skoru: {difficulty_score:.2f} (boş hücre sayısı)")
    print(f"{'='*60}")
    
    # Tensor to numpy
    if torch.is_tensor(puzzle):
        puzzle = puzzle.cpu().numpy()
    if torch.is_tensor(solution):
        solution = solution.cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    
    puzzle = puzzle.reshape(9, 9)
    solution = solution.reshape(9, 9)
    prediction = prediction.reshape(9, 9)
    
    print("\n🔹 Başlangıç Sudoku:")
    print_sudoku_grid(puzzle)
    
    print("\n🤖 Model Çözümü:")
    error_positions = print_sudoku_grid(prediction, solution if not is_correct else None)
    
    if not is_correct:
        print("\n✅ Doğru Çözüm:")
        print_sudoku_grid(solution)
        
        # Hata analizi
        num_errors = np.sum(prediction != solution)
        print(f"\n📊 Hata Analizi:")
        print(f"   • Toplam hatalı hücre: {num_errors}/81")
        print(f"   • Doğruluk oranı: {((81-num_errors)/81)*100:.1f}%")
        
        # Hangi bölgelerde hata var?
        print(f"\n🔍 Hatalı Bölgeler:")
        analyze_error_regions(prediction, solution)

def print_sudoku_grid(grid, highlight_errors=None):
    """Sudoku grid'ini güzel formatta yazdır"""
    error_positions = []
    print("    " + "─" * 25)
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("    " + "─" * 25)
        row = f" {i+1}  "
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row += "│"
            val = grid[i, j]
            display_val = "·" if val == 1 else str(int(val) % 10)
            
            # Hataları vurgula
            if highlight_errors is not None and grid[i, j] != highlight_errors[i, j]:
                row += f" \033[91m{display_val}\033[0m"  # Kırmızı
                error_positions.append((i, j))
            else:
                row += f" {display_val}"
        print(row)
    print("    " + "─" * 25)
    print("      1 2 3 4 5 6 7 8 9")
    return error_positions

def analyze_error_regions(prediction, solution):
    """Hataların hangi bölgelerde olduğunu analiz et"""
    # Satır bazında hatalar
    row_errors = defaultdict(int)
    col_errors = defaultdict(int)
    box_errors = defaultdict(int)
    
    for i in range(9):
        for j in range(9):
            if prediction[i, j] != solution[i, j]:
                row_errors[i] += 1
                col_errors[j] += 1
                box_idx = (i // 3) * 3 + (j // 3)
                box_errors[box_idx] += 1
    
    if row_errors:
        print(f"   • Satırlardaki hatalar: {dict([(k+1, v) for k, v in row_errors.items()])}")
    if col_errors:
        print(f"   • Sütunlardaki hatalar: {dict([(k+1, v) for k, v in col_errors.items()])}")
    if box_errors:
        print(f"   • 3x3 kutulardaki hatalar: {dict([(k+1, v) for k, v in box_errors.items()])}")

def calculate_difficulty(puzzle):
    """Sudoku zorluğunu hesapla (boş hücre sayısı)"""
    if torch.is_tensor(puzzle):
        puzzle = puzzle.cpu().numpy()
    return np.sum(puzzle.reshape(-1) == 1)

def generate_html_report(results, output_path):
    """HTML rapor oluştur"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Sudoku Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
            .stat-card h3 { margin: 0; font-size: 14px; opacity: 0.9; }
            .stat-card p { margin: 10px 0 0 0; font-size: 32px; font-weight: bold; }
            .sudoku-grid { display: grid; grid-template-columns: repeat(9, 40px); gap: 1px; background: #333; margin: 10px 0; width: fit-content; }
            .sudoku-cell { width: 40px; height: 40px; background: white; display: flex; align-items: center; justify-content: center; font-weight: bold; }
            .error { background: #ff6b6b; color: white; }
            .correct { background: #51cf66; color: white; }
            .given { background: #e9ecef; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #3498db; color: white; }
            tr:hover { background: #f5f5f5; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Sudoku Model Evaluation Report</h1>
    """
    
    html += f"""
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Puzzles</h3>
                    <p>{results['total']}</p>
                </div>
                <div class="stat-card">
                    <h3>Correct Solutions</h3>
                    <p>{results['correct']}</p>
                </div>
                <div class="stat-card">
                    <h3>Accuracy</h3>
                    <p>{results['accuracy']:.1f}%</p>
                </div>
                <div class="stat-card">
                    <h3>Avg Difficulty</h3>
                    <p>{results['avg_difficulty']:.1f}</p>
                </div>
            </div>
            
            <h2>📊 Detailed Results</h2>
            <table>
                <tr>
                    <th>#</th>
                    <th>Status</th>
                    <th>Difficulty</th>
                    <th>Errors</th>
                    <th>Cell Accuracy</th>
                </tr>
    """
    
    for item in results['details']:
        status = "✅ Correct" if item['correct'] else "❌ Wrong"
        html += f"""
                <tr>
                    <td>{item['idx']}</td>
                    <td>{status}</td>
                    <td>{item['difficulty']}</td>
                    <td>{item['errors']}/81</td>
                    <td>{item['cell_accuracy']:.1f}%</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n📄 HTML rapor kaydedildi: {output_path}")

def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))
    
    RANK = 0
    WORLD_SIZE = 1
    
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)
    
    print(f"📊 Loading datasets...")
    train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1, 
        global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE
    )
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1, 
        global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE
    )
    
    eval_loader_list = list(eval_loader)
    if eval_cfg.num_samples is not None:
        eval_loader_list = eval_loader_list[:eval_cfg.num_samples]
        print(f"🎯 Testing on {len(eval_loader_list)} batches (limited to {eval_cfg.num_samples})")
    else:
        print(f"🎯 Testing on {len(eval_loader_list)} batches (full dataset)")
    
    print(f"🤖 Loading model from checkpoint...")
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    
    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location="cuda"), assign=True)
    except:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location="cuda").items()}, 
            assign=True
        )
    
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))
    
    print("🚀 Starting evaluation...")
    train_state.model.eval()
    
    metrics = evaluate(config, train_state, eval_loader_list, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
    
    if metrics is not None:
        print("\n" + "="*70)
        print("📈 EVALUATION RESULTS")
        print("="*70)
        for key, value in metrics['all'].items():
            if isinstance(value, (np.floating, float)):
                if 'accuracy' in key:
                    print(f"  {key:25s}: {value*100:6.2f}%")
                else:
                    print(f"  {key:25s}: {value:6.4f}")
            else:
                print(f"  {key:25s}: {value}")
        print("="*70)
    
    # Detailed analysis and visualization
    if (eval_cfg.visualize or eval_cfg.detailed_stats or eval_cfg.save_html_report) and RANK == 0:
        output_file = os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{RANK}")
        
        if os.path.exists(output_file):
            print(f"\n🔍 Loading predictions from {output_file}...")
            data = torch.load(output_file, map_location='cpu')
            
            num_samples = len(data['inputs'])
            correct_puzzles = []
            wrong_puzzles = []
            
            # Collect statistics
            stats = {
                'total': num_samples,
                'correct': 0,
                'difficulties': [],
                'details': []
            }
            
            for i in range(num_samples):
                inputs = data['inputs'][i]
                labels = data['labels'][i]
                preds = data['logits'][i].argmax(-1)
                
                is_correct = torch.all(preds == labels).item()
                difficulty = calculate_difficulty(inputs)
                num_errors = torch.sum(preds != labels).item()
                cell_accuracy = ((81 - num_errors) / 81) * 100
                
                stats['difficulties'].append(difficulty)
                stats['details'].append({
                    'idx': i + 1,
                    'correct': is_correct,
                    'difficulty': difficulty,
                    'errors': num_errors,
                    'cell_accuracy': cell_accuracy
                })
                
                if is_correct:
                    stats['correct'] += 1
                    correct_puzzles.append((i, inputs, labels, preds, difficulty))
                else:
                    wrong_puzzles.append((i, inputs, labels, preds, difficulty))
            
            stats['accuracy'] = (stats['correct'] / stats['total']) * 100
            stats['avg_difficulty'] = np.mean(stats['difficulties'])
            
            # Print detailed statistics
            if eval_cfg.detailed_stats:
                print("\n" + "="*70)
                print("📊 DETAILED STATISTICS")
                print("="*70)
                print(f"  Total Puzzles:          {stats['total']}")
                print(f"  Correct Solutions:      {stats['correct']} ({stats['accuracy']:.2f}%)")
                print(f"  Wrong Solutions:        {stats['total'] - stats['correct']}")
                print(f"  Average Difficulty:     {stats['avg_difficulty']:.2f} empty cells")
                print(f"  Min Difficulty:         {min(stats['difficulties'])}")
                print(f"  Max Difficulty:         {max(stats['difficulties'])}")
                print("="*70)
                
                # Zorluk bazında başarı oranı
                if stats['difficulties']:
                    print("\n🔥 Başarı Oranı - Zorluk Seviyesine Göre:")
                    difficulty_ranges = [(0, 40), (40, 50), (50, 60), (60, 81)]
                    for low, high in difficulty_ranges:
                        in_range = [d for idx, d in enumerate(stats['difficulties']) 
                                   if low <= d < high and stats['details'][idx]['correct']]
                        total_in_range = len([d for d in stats['difficulties'] if low <= d < high])
                        if total_in_range > 0:
                            success_rate = (len(in_range) / total_in_range) * 100
                            print(f"  {low:2d}-{high:2d} boş hücre: {success_rate:5.1f}% ({len(in_range)}/{total_in_range})")
            
            # Visualize examples
            if eval_cfg.visualize:
                puzzles_to_show = []
                
                if eval_cfg.show_only_errors:
                    puzzles_to_show = wrong_puzzles[:eval_cfg.visualize_count]
                    print(f"\n🎨 Showing {len(puzzles_to_show)} WRONG solutions...")
                else:
                    # Doğru ve yanlışları karışık göster
                    num_correct = min(eval_cfg.visualize_count // 2, len(correct_puzzles))
                    num_wrong = min(eval_cfg.visualize_count - num_correct, len(wrong_puzzles))
                    puzzles_to_show = correct_puzzles[:num_correct] + wrong_puzzles[:num_wrong]
                    print(f"\n🎨 Visualizing {len(puzzles_to_show)} examples ({num_correct} correct, {num_wrong} wrong)...")
                
                for idx, inputs, labels, preds, difficulty in puzzles_to_show:
                    is_correct = torch.all(preds == labels).item()
                    visualize_sudoku(inputs, labels, preds, idx, is_correct, difficulty)
            
            # Generate HTML report
            if eval_cfg.save_html_report:
                report_path = os.path.join(config.checkpoint_path, "evaluation_report.html")
                generate_html_report(stats, report_path)
        else:
            print(f"⚠️  Output file not found: {output_file}")

if __name__ == "__main__":
    launch()