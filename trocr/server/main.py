import ray
from ray import serve
from trocr_server import TrOCRService
import asyncio

async def run_server():
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    
    serve.run(
        TrOCRService.bind(),
        name="trocr",
        route_prefix="/trocr"
    )

    print("Сервер запущен. Доступно на http://localhost:8000/trocr")
    
    # Ожидаем завершения (можно добавить условия выхода)
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nЗавершение работы сервера...")
        serve.shutdown()
        ray.shutdown()