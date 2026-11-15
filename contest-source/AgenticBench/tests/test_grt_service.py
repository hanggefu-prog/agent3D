"""Real loopback broker protocol tests; no EDA or Docker dependency."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench.service import create_server
from grt_bench.queue import Queue
from grt_bench.worker import Client


class BrokerTests(unittest.TestCase):
    def test_release_protocol_frees_gpu_but_preserves_live_cpu_job(self):
        with tempfile.TemporaryDirectory() as directory:
            queue=Queue(Path(directory)/"queue.sqlite")
            queue.register_host("worker",dict(cpu=8,memory_gb=16,storage_gb=20,gpus=["0"],build_slots=1))
            resources=dict(cpu=4,memory_gb=8,storage_gb=5,gpu_count=1,build_slots=0)
            first=queue.submit({"cell":1},resources)
            second=queue.submit({"cell":2},resources)
            server=create_server(queue.path,{"worker":"test-token","other":"other-token"},0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                url="http://127.0.0.1:"+str(server.server_address[1])
                client=Client(url,"worker","test-token")
                lease=client.call("claim",{})
                self.assertEqual(lease["id"],first)
                args=dict(job=first,token=lease["token"],resources=dict(resources,cpu=2,memory_gb=4,gpu_count=0),
                    evidence=dict(terminated=True,container="grt-"+first[:32]+"-frontend",checkpoint_sha256="a"*64))
                with self.assertRaises(HTTPError): Client(url,"other","other-token").call("release",args)
                with self.assertRaises(HTTPError): client.call("release",dict(args,token="stale"))
                allocation=client.call("release",args)
                self.assertEqual(allocation["gpus"],[])
                self.assertEqual(client.call("claim",{})["id"],second)
                client.call("heartbeat",dict(job=first,token=lease["token"]))
                client.call("finish",dict(job=first,token=lease["token"],success=True,result={"fixture":True}))
                self.assertEqual(Queue(queue.path).rows()[0]["state"],"succeeded")
            finally:
                server.shutdown();server.server_close();thread.join(timeout=2)

    def test_authenticated_lease_and_cancel_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            database=Path(directory)/"queue.sqlite"
            queue=Queue(database)
            queue.register_host("worker",dict(cpu=4,memory_gb=8,storage_gb=10,gpus=["0"],build_slots=1))
            job=queue.submit({"fixture":True},dict(cpu=2,memory_gb=4,storage_gb=5,gpu_count=1,build_slots=0))
            server=create_server(database,{"worker":"test-token"},0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                url="http://127.0.0.1:"+str(server.server_address[1])
                with self.assertRaises(HTTPError) as denied:
                    Client(url,"worker","wrong-token").call("claim",{})
                self.assertEqual(denied.exception.code,403)
                client=Client(url,"worker","test-token")
                lease=client.call("claim",{})
                self.assertEqual(lease["id"],job)
                self.assertEqual(lease["gpus"],["0"])
                self.assertIsNone(client.call("claim",{}))
                client.call("heartbeat",{"job":job,"token":lease["token"]})
                queue.cancel(job)
                with self.assertRaises(HTTPError) as cancelled:
                    client.call("heartbeat",{"job":job,"token":lease["token"]})
                self.assertEqual(cancelled.exception.code,409)
                self.assertEqual(Queue(database).rows()[0]["state"],"cancelling")
                with self.assertRaises(HTTPError):
                    client.call("register-host",{})
            finally:
                server.shutdown();server.server_close();thread.join(timeout=2)


if __name__=="__main__":unittest.main()
