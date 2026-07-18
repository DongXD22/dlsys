"""带超时的测试运行器，输出详细信息"""
import subprocess
import sys
import os

def run_test(query, timeout=10, cwd=None):
    """运行 pytest 测试，带超时和详细输出"""
    if cwd is None:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    cmd = [sys.executable, '-m', 'pytest', '-v', '-s', '-k', query, '--tb=long', '--no-header']
    
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        print("=== STDOUT ===")
        print(result.stdout)
        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)
        print(f"=== Return code: {result.returncode} ===")
        return result.returncode
    except subprocess.TimeoutExpired as e:
        print(f"!!! TIMEOUT ({timeout}s) !!!")
        if e.stdout:
            print("=== STDOUT (before timeout) ===")
            print(e.stdout)
        if e.stderr:
            print("=== STDERR (before timeout) ===")
            print(e.stderr)
        return -1

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "compact and cpu"
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_test(query, timeout)
