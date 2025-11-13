#!/bin/bash
# =============================================================================
# 🧪 HRM-FREE-META: Comprehensive Configuration Testing
# Tests all 4 PRESET configurations to verify model can learn in each mode
# =============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "🧪 HRM-FREE-META COMPREHENSIVE TESTING"
echo "================================================================================"
echo ""

# Create test results directory
mkdir -p test_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="test_results/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Test parameters (quick tests for validation)
QUICK_EPOCHS=5
QUICK_BATCH_SIZE=16
TEST_DATA="data/sudoku-extreme-1k-aug-1000"

# =============================================================================
# TEST 1: Standard HRM Pretraining (PRESET 1)
# =============================================================================
echo "================================================================================"
echo "TEST 1: Standard HRM Pretraining (Original cfg_pretrain.yaml)"
echo "================================================================================"
echo "Expected: Standard ACT-based training should work"
echo "Key metrics: train/loss should decrease, train/accuracy should increase"
echo ""

python unified_training.py \
    mode=pretrain \
    arch.name="hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1" \
    arch.loss.name="losses@ACTLossHead" \
    data_path="$TEST_DATA" \
    global_batch_size="$QUICK_BATCH_SIZE" \
    epochs="$QUICK_EPOCHS" \
    eval_interval=null \
    wandb_enabled=false \
    checkpoint_path="$LOG_DIR/test1_standard_hrm" \
    2>&1 | tee "$LOG_DIR/test1_standard_hrm.log"

echo ""
echo "✅ TEST 1 COMPLETE"
echo "Check: $LOG_DIR/test1_standard_hrm.log"
echo ""
sleep 2

# =============================================================================
# TEST 2: HRM-Free-Meta Pretraining (PRESET 2)
# =============================================================================
echo "================================================================================"
echo "TEST 2: HRM-Free-Meta Pretraining (cfg_pretrain_meta.yaml)"
echo "================================================================================"
echo "Expected: Meta-enhanced model should learn with standard supervision"
echo "Key metrics: train/meta/loss_task should decrease, z_context_norm should stabilize"
echo ""

python unified_training.py \
    mode=pretrain \
    arch.name="hrm.hrm_free_meta@HRMFreeMeta" \
    arch.loss.name="meta_loss_head@MetaLearningLossHead" \
    data_path="$TEST_DATA" \
    global_batch_size="$QUICK_BATCH_SIZE" \
    epochs="$QUICK_EPOCHS" \
    eval_interval=null \
    wandb_enabled=false \
    arch.alpha_kl=0.001 \
    arch.beta_entropy=0.01 \
    checkpoint_path="$LOG_DIR/test2_meta_pretrain" \
    2>&1 | tee "$LOG_DIR/test2_meta_pretrain.log"

echo ""
echo "✅ TEST 2 COMPLETE"
echo "Check: $LOG_DIR/test2_meta_pretrain.log"
echo ""
sleep 2

# =============================================================================
# TEST 3: Meta-Learning Training (PRESET 3)
# =============================================================================
echo "================================================================================"
echo "TEST 3: Meta-Learning Training (meta_training.yaml)"
echo "================================================================================"
echo "Expected: MAML-style meta-learning should work"
echo "Key metrics: meta_loss and query_accuracy should improve"
echo ""

python unified_training.py \
    mode=meta \
    arch.name="hrm.hrm_free_meta@HRMFreeMeta" \
    arch.loss.name="meta_loss_head@MetaLearningLossHead" \
    data_path="$TEST_DATA" \
    global_batch_size=8 \
    epochs="$QUICK_EPOCHS" \
    eval_interval=null \
    wandb_enabled=false \
    meta.enabled=true \
    meta.n_support=5 \
    meta.n_query=10 \
    meta.task_batch_size=4 \
    meta.inner_steps=5 \
    meta.inner_lr=0.001 \
    meta.meta_batches_per_epoch=20 \
    checkpoint_path="$LOG_DIR/test3_meta_learning" \
    2>&1 | tee "$LOG_DIR/test3_meta_learning.log"

echo ""
echo "✅ TEST 3 COMPLETE"
echo "Check: $LOG_DIR/test3_meta_learning.log"
echo ""
sleep 2

# =============================================================================
# TEST 4: Laptop/Limited GPU Config (PRESET 4)
# =============================================================================
echo "================================================================================"
echo "TEST 4: Laptop/Limited GPU Config"
echo "================================================================================"
echo "Expected: Smaller model should train faster with less memory"
echo "Key metrics: Should complete quickly, similar learning patterns"
echo ""

python unified_training.py \
    mode=pretrain \
    arch.name="hrm.hrm_free_meta@HRMFreeMeta" \
    arch.loss.name="meta_loss_head@MetaLearningLossHead" \
    data_path="$TEST_DATA" \
    global_batch_size=8 \
    epochs="$QUICK_EPOCHS" \
    eval_interval=null \
    wandb_enabled=false \
    arch.hidden_size=128 \
    arch.H_layers=2 \
    arch.L_layers=2 \
    arch.z_dim=32 \
    arch.encoder_layers=1 \
    arch.controller_hidden=64 \
    arch.puzzle_emb_ndim=128 \
    checkpoint_path="$LOG_DIR/test4_laptop_config" \
    2>&1 | tee "$LOG_DIR/test4_laptop_config.log"

echo ""
echo "✅ TEST 4 COMPLETE"
echo "Check: $LOG_DIR/test4_laptop_config.log"
echo ""
sleep 2

# =============================================================================
# TEST 5: Meta-Learning with Small Model (Bonus)
# =============================================================================
echo "================================================================================"
echo "TEST 5: Meta-Learning with Small Model (Stress Test)"
echo "================================================================================"
echo "Expected: Meta-learning should work even with tiny model"
echo ""

python unified_training.py \
    mode=meta \
    arch.name="hrm.hrm_free_meta@HRMFreeMeta" \
    arch.loss.name="meta_loss_head@MetaLearningLossHead" \
    data_path="$TEST_DATA" \
    global_batch_size=8 \
    epochs=3 \
    eval_interval=null \
    wandb_enabled=false \
    meta.enabled=true \
    meta.n_support=3 \
    meta.n_query=5 \
    meta.task_batch_size=2 \
    meta.inner_steps=3 \
    meta.meta_batches_per_epoch=10 \
    arch.hidden_size=128 \
    arch.H_layers=1 \
    arch.L_layers=1 \
    arch.z_dim=32 \
    arch.puzzle_emb_ndim=128 \
    checkpoint_path="$LOG_DIR/test5_meta_small" \
    2>&1 | tee "$LOG_DIR/test5_meta_small.log"

echo ""
echo "✅ TEST 5 COMPLETE"
echo ""

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================
echo ""
echo "================================================================================"
echo "📊 TEST RESULTS SUMMARY"
echo "================================================================================"
echo ""
echo "All test logs saved to: $LOG_DIR"
echo ""

# Check each test for success
check_test() {
    local test_num=$1
    local test_name=$2
    local log_file="$LOG_DIR/test${test_num}_*.log"
    
    if grep -q "TRAINING COMPLETE" $log_file 2>/dev/null; then
        echo "✅ TEST $test_num: $test_name - PASSED"
        
        # Extract key metrics
        if [ "$test_num" = "1" ]; then
            echo "   └─ Final loss: $(grep 'train/loss:' $log_file | tail -1 | awk '{print $NF}')"
        elif [ "$test_num" = "2" ]; then
            echo "   └─ Meta loss: $(grep 'meta/total_loss:' $log_file | tail -1 | awk '{print $NF}')"
        elif [ "$test_num" = "3" ] || [ "$test_num" = "5" ]; then
            echo "   └─ Query acc: $(grep 'Query Acc:' $log_file | tail -1 | awk '{print $NF}')"
        fi
    else
        echo "❌ TEST $test_num: $test_name - FAILED"
        echo "   └─ Check log: $log_file"
    fi
}

check_test 1 "Standard HRM"
check_test 2 "Meta Pretrain"
check_test 3 "Meta Learning"
check_test 4 "Laptop Config"
check_test 5 "Meta Small"

echo ""
echo "================================================================================"
echo "📈 DETAILED ANALYSIS"
echo "================================================================================"
echo ""
echo "To analyze results in detail:"
echo ""
echo "  # View specific test log"
echo "  cat $LOG_DIR/test1_standard_hrm.log"
echo ""
echo "  # Compare final losses"
echo "  grep 'loss:' $LOG_DIR/*.log | tail -5"
echo ""
echo "  # Check model sizes"
echo "  grep 'parameters' $LOG_DIR/*.log"
echo ""
echo "  # Examine checkpoints"
echo "  ls -lh $LOG_DIR/test*/checkpoint_*.pt"
echo ""
echo "================================================================================"
echo "✅ ALL TESTS COMPLETE!"
echo "================================================================================"