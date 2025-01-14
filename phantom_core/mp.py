import traceback
import sys
import os
import subprocess


class ExceptionWithTraceback(Exception):

    @classmethod
    def from_exception(cls, e: Exception):
        
        if sys.exc_info()[0] is not None:
            tb = traceback.format_exc()
        else:
            tb = None

        _msg = f"{type(e).__name__}({e})"

        if tb is not None:
            _msg += f"\n{tb}"

        return cls(_msg)



def kill_process_on_port(port: int):
    try:
        # Find the process using the port
        result = subprocess.check_output(f"lsof -i :{port} -t", shell=True)
        pids = result.decode().strip().split('\n')
        
        # Kill each process
        for pid in pids:
            os.kill(int(pid), 9)
        print(f"Killed processes on port {port}")
    except subprocess.CalledProcessError:
        print(f"No process is using port {port}")