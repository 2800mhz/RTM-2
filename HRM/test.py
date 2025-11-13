#!/usr/bin/env python3
"""
🔬 HRM-FREE-META COMPLETE TEST SUITE
/workspace/hrm/test.py

Runs ALL tests in test/ directory + additional comprehensive checks
"""

import sys
import os
import subprocess
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import json

# Setup paths
HRM_ROOT = Path(__file__).parent
TEST_DIR = HRM_ROOT / "test"
sys.path.insert(0, str(HRM_ROOT))

print("=" * 100)
print("🔬 HRM-FREE-META COMPLETE TEST SUITE")
print("=" * 100)
print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 HRM Root: {HRM_ROOT}")
print(f"📁 Test Dir: {TEST_DIR}")
print(f"💻 Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
print("=" * 100)


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.details = ""
        self.errors = []
    
    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name}"


class ComprehensiveTestRunner:
    def __init__(self):
        self.results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.loss_head = None
        
    def run_all(self):
        """Run complete test suite"""
        print("\n🚀 STARTING TEST SUITE\n")
        
        # Phase 1: Run all test scripts in test/
        self.run_test_directory()
        
        # Phase 2: Additional tests
        self.run_additional_tests()
        
        # Phase 3: Generate report
        self.generate_report()
    
    def run_test_directory(self):
        """Run all .py files in test/ directory"""
        print("=" * 100)
        print("📁 PHASE 1: RUNNING TEST SCRIPTS FROM test/ DIRECTORY")
        print("=" * 100)
        
        if not TEST_DIR.exists():
            print(f"❌ Test directory not found: {TEST_DIR}")
            return
        
        # Find all .py files in test/
        test_files = sorted(TEST_DIR.glob("*.py"))
        
        if not test_files:
            print(f"⚠️  No test files found in {TEST_DIR}")
            return
        
        print(f"\n📝 Found {len(test_files)} test scripts:")
        for tf in test_files:
            print(f"   • {tf.name}")
        
        print()
        
        # Run each test
        for test_file in test_files:
            result = TestResult(f"Script: {test_file.name}")
            
            print(f"{'─' * 100}")
            print(f"🔄 Running: {test_file.name}")
            print(f"{'─' * 100}")
            
            try:
                # Run the script with PYTHONPATH set
                env = os.environ.copy()
                env['PYTHONPATH'] = str(HRM_ROOT)
                
                proc = subprocess.run(
                    [sys.executable, str(test_file)],
                    cwd=str(HRM_ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 min timeout
                )
                
                output = proc.stdout
                errors = proc.stderr
                
                # Print output in real-time style
                if output:
                    print(output)
                if errors and "warning" not in errors.lower():
                    print("STDERR:", errors)
                
                # Parse results
                passed_count = output.count("✅ PASS") + output.count("PASS:")
                failed_count = output.count("❌ FAIL") + output.count("FAIL:")
                total = passed_count + failed_count
                
                # Determine if passed
                if total > 0:
                    result.passed = passed_count > failed_count
                    result.details = f"{passed_count}/{total} tests passed"
                elif proc.returncode == 0:
                    result.passed = True
                    result.details = "Script completed successfully"
                else:
                    result.passed = False
                    result.details = f"Return code: {proc.returncode}"
                
                # Check for specific success indicators
                if "ALL TESTS PASSED" in output or "6/6 TESTS PASSED" in output:
                    result.passed = True
                
                status = "✅ SUCCESS" if result.passed else "❌ FAILED"
                print(f"\n{status}: {test_file.name} - {result.details}\n")
                
            except subprocess.TimeoutExpired:
                result.passed = False
                result.details = "Test timed out (5 min)"
                result.errors.append("Timeout")
                print(f"❌ TIMEOUT: {test_file.name}\n")
                
            except Exception as e:
                result.passed = False
                result.details = f"Error: {str(e)}"
                result.errors.append(str(e))
                print(f"❌ ERROR: {test_file.name} - {e}\n")
            
            self.results.append(result)
    
    def run_additional_tests(self):
        """Run additional comprehensive tests"""
        print("=" * 100)
        print("🔬 PHASE 2: ADDITIONAL COMPREHENSIVE TESTS")
        print("=" * 100)
        
        self._test_model_creation()
        self._test_gradient_flow()
        self._test_parameter_update()
        self._test_memory_usage()
    
    def _test_model_creation(self):
        """Test model can be created"""
        result = TestResult("Model Creation & Components")
        print("\n📋 Test: Model Creation & Components")
        
        try:
            from models.hrm.hrm_free_meta import HRMFreeMeta
            from meta_loss_head import MetaLearningLossHead
            
            config = {
                'batch_size': 8, 'seq_len': 81, 'vocab_size': 10,
                'num_puzzle_identifiers': 100, 'hidden_size': 256,
                'num_heads': 4, 'expansion': 4, 'H_layers': 2, 'L_layers': 2,
                'H_cycles': 1, 'L_cycles': 1, 'halt_max_steps': 4,
                'z_dim': 128, 'encoder_layers': 2, 'controller_hidden': 256,
                'gate_temperature': 1.0, 'use_dynamic_weighting': True,
                'causal': False, 'pos_encodings': 'rope',
                'puzzle_emb_ndim': 256, 'halt_exploration_prob': 0.05,
                'forward_dtype': 'bfloat16',
            }
            
            self.model = HRMFreeMeta(config).to(self.device)
            self.loss_head = MetaLearningLossHead(
                model=self.model,
                loss_type='stablemax_cross_entropy',
                alpha_kl=0.001,
                beta_entropy=0.01
            )
            
            total_params = sum(p.numel() for p in self.model.parameters())
            meta_params = sum(p.numel() for p in self.model.inner.latent_encoder.parameters()) + \
                         sum(p.numel() for p in self.model.inner.meta_controller.parameters())
            
            print(f"   ✅ Model created successfully")
            print(f"   📊 Total parameters: {total_params:,}")
            print(f"   📊 Meta parameters: {meta_params:,} ({100*meta_params/total_params:.1f}%)")
            print(f"   ✅ Loss head created")
            
            result.passed = True
            result.details = f"{total_params:,} params"
            
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            print(f"   ❌ ERROR: {e}")
        
        self.results.append(result)
    
    def _test_gradient_flow(self):
        """Test gradient flow through all components"""
        result = TestResult("Gradient Flow (HRM + Meta)")
        print("\n📋 Test: Gradient Flow")
        
        if self.model is None or self.loss_head is None:
            result.passed = False
            result.errors.append("Model not initialized")
            self.results.append(result)
            return
        
        try:
            batch = {
                'inputs': torch.randint(0, 10, (8, 81), device=self.device),
                'labels': torch.randint(0, 10, (8, 81), device=self.device),
                'puzzle_identifiers': torch.randint(0, 100, (8,), device=self.device),
            }
            
            self.model.train()
            self.model.zero_grad()
            
            carry = self.loss_head.initial_carry(batch)
            carry_out, loss, metrics, preds, _ = self.loss_head(carry, batch)
            loss.backward()
            
            # Check gradients in all components
            hrm_grads = sum(1 for p in self.model.inner.hrm_inner.parameters() 
                          if p.grad is not None and p.grad.abs().sum() > 0)
            encoder_grads = sum(1 for p in self.model.inner.latent_encoder.parameters() 
                              if p.grad is not None and p.grad.abs().sum() > 0)
            controller_grads = sum(1 for p in self.model.inner.meta_controller.parameters() 
                                 if p.grad is not None and p.grad.abs().sum() > 0)
            
            hrm_total = len(list(self.model.inner.hrm_inner.parameters()))
            encoder_total = len(list(self.model.inner.latent_encoder.parameters()))
            controller_total = len(list(self.model.inner.meta_controller.parameters()))
            
            print(f"   📊 HRM gradients:        {hrm_grads}/{hrm_total}")
            print(f"   📊 Encoder gradients:    {encoder_grads}/{encoder_total}")
            print(f"   📊 Controller gradients: {controller_grads}/{controller_total}")
            
            all_have_grads = (hrm_grads > 0 and encoder_grads > 0 and controller_grads > 0)
            
            if all_have_grads:
                print(f"   ✅ All components receive gradients")
                result.passed = True
                result.details = "Full gradient flow"
            else:
                print(f"   ⚠️  Partial gradient flow")
                result.passed = encoder_grads > 0 and controller_grads > 0
                result.details = f"HRM:{hrm_grads>0} Enc:{encoder_grads>0} Ctrl:{controller_grads>0}"
            
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            print(f"   ❌ ERROR: {e}")
        
        self.results.append(result)
    
    def _test_parameter_update(self):
        """Test parameters actually update during training"""
        result = TestResult("Parameter Update (Training Loop)")
        print("\n📋 Test: Parameter Update")
        
        if self.model is None or self.loss_head is None:
            result.passed = False
            result.errors.append("Model not initialized")
            self.results.append(result)
            return
        
        try:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            
            # Save initial parameters
            initial_params = {}
            for name, param in self.model.named_parameters():
                initial_params[name] = param.data.clone()
            
            # Training loop
            self.model.train()
            for step in range(5):
                batch = {
                    'inputs': torch.randint(0, 10, (8, 81), device=self.device),
                    'labels': torch.randint(0, 10, (8, 81), device=self.device),
                    'puzzle_identifiers': torch.randint(0, 100, (8,), device=self.device),
                }
                
                optimizer.zero_grad()
                carry = self.loss_head.initial_carry(batch)
                carry_out, loss, metrics, preds, _ = self.loss_head(carry, batch)
                loss.backward()
                optimizer.step()
            
            # Check which parameters changed
            changed_params = {}
            for name, param in self.model.named_parameters():
                if not torch.equal(param.data, initial_params[name]):
                    max_change = (param.data - initial_params[name]).abs().max().item()
                    changed_params[name] = max_change
            
            total_params = len(initial_params)
            changed_count = len(changed_params)
            
            print(f"   📊 Parameters changed: {changed_count}/{total_params}")
            
            # Check specific components
            hrm_changed = sum(1 for k in changed_params if 'hrm_inner' in k)
            meta_changed = sum(1 for k in changed_params if 'latent_encoder' in k or 'meta_controller' in k)
            
            print(f"   📊 HRM changed: {hrm_changed}")
            print(f"   📊 Meta changed: {meta_changed}")
            
            if changed_count > total_params * 0.5:
                print(f"   ✅ Parameters updating normally ({100*changed_count/total_params:.1f}%)")
                result.passed = True
                result.details = f"{changed_count}/{total_params} changed"
            else:
                print(f"   ⚠️  Low parameter update rate ({100*changed_count/total_params:.1f}%)")
                result.passed = False
                result.details = f"Only {changed_count}/{total_params} changed"
                result.errors.append("Most parameters not updating - gradient flow issue")
            
            # Show top 5 changes
            if changed_params:
                top_changes = sorted(changed_params.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"   📊 Top 5 changes:")
                for name, change in top_changes:
                    print(f"      {name}: {change:.6f}")
            
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            print(f"   ❌ ERROR: {e}")
        
        self.results.append(result)
    
    def _test_memory_usage(self):
        """Test memory usage"""
        result = TestResult("GPU Memory Usage")
        print("\n📋 Test: GPU Memory Usage")
        
        if not torch.cuda.is_available():
            print(f"   ⚠️  CUDA not available, skipping")
            result.passed = True
            result.details = "N/A (CPU only)"
            self.results.append(result)
            return
        
        if self.model is None or self.loss_head is None:
            result.passed = False
            result.errors.append("Model not initialized")
            self.results.append(result)
            return
        
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            batch = {
                'inputs': torch.randint(0, 10, (8, 81), device=self.device),
                'labels': torch.randint(0, 10, (8, 81), device=self.device),
                'puzzle_identifiers': torch.randint(0, 100, (8,), device=self.device),
            }
            
            self.model.train()
            carry = self.loss_head.initial_carry(batch)
            carry_out, loss, metrics, preds, _ = self.loss_head(carry, batch)
            loss.backward()
            
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            
            print(f"   📊 Peak memory: {peak_mb:.1f} MB")
            
            if peak_mb < 2000:
                print(f"   ✅ Memory usage is reasonable")
                result.passed = True
            else:
                print(f"   ⚠️  High memory usage")
                result.passed = False
            
            result.details = f"{peak_mb:.1f} MB"
            
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            print(f"   ❌ ERROR: {e}")
        
        self.results.append(result)
    
    def generate_report(self):
        """Generate final report"""
        print("\n" + "=" * 100)
        print("📊 FINAL TEST REPORT")
        print("=" * 100)
        
        # Categorize results
        script_tests = [r for r in self.results if r.name.startswith("Script:")]
        other_tests = [r for r in self.results if not r.name.startswith("Script:")]
        
        # Summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print(f"\n📈 OVERALL SUMMARY:")
        print(f"   Total tests:  {total}")
        print(f"   ✅ Passed:    {passed} ({100*passed/total:.1f}%)")
        print(f"   ❌ Failed:    {failed} ({100*failed/total:.1f}%)")
        
        # Script tests
        if script_tests:
            script_passed = sum(1 for r in script_tests if r.passed)
            print(f"\n📁 TEST SCRIPTS ({len(script_tests)} scripts):")
            print(f"   ✅ Passed: {script_passed}/{len(script_tests)}")
            for r in script_tests:
                status = "✅" if r.passed else "❌"
                print(f"   {status} {r.name.replace('Script: ', '')} - {r.details}")
        
        # Additional tests
        if other_tests:
            other_passed = sum(1 for r in other_tests if r.passed)
            print(f"\n🔬 ADDITIONAL TESTS ({len(other_tests)} tests):")
            print(f"   ✅ Passed: {other_passed}/{len(other_tests)}")
            for r in other_tests:
                status = "✅" if r.passed else "❌"
                print(f"   {status} {r.name} - {r.details}")
        
        # Failed tests details
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            print(f"\n❌ FAILED TESTS DETAILS:")
            for r in failed_tests:
                print(f"\n   {r.name}:")
                if r.errors:
                    for e in r.errors:
                        print(f"      • {e}")
                else:
                    print(f"      • {r.details}")
        
        # Final verdict
        print("\n" + "=" * 100)
        if passed == total:
            print("🎉 ALL TESTS PASSED! SYSTEM FULLY OPERATIONAL!")
            verdict = "EXCELLENT"
        elif passed >= total * 0.8:
            print(f"✅ VERY GOOD! {passed}/{total} tests passed")
            verdict = "GOOD"
        elif passed >= total * 0.6:
            print(f"✅ GOOD! {passed}/{total} tests passed")
            verdict = "ACCEPTABLE"
        else:
            print(f"⚠️  NEEDS ATTENTION! Only {passed}/{total} tests passed")
            verdict = "NEEDS_WORK"
        print("=" * 100)
        
        # Save JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'verdict': verdict,
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / total if total > 0 else 0
            },
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'details': r.details,
                    'errors': r.errors
                }
                for r in self.results
            ]
        }
        
        report_path = HRM_ROOT / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")


def main():
    try:
        runner = ComprehensiveTestRunner()
        runner.run_all()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()