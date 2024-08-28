import litserve as ls

from fastapi.responses import JSONResponse
from fastapi.requests import Request


class InvalidInputError(Exception):
    pass


async def invalid_input_error_handler(request: Request, exc: InvalidInputError):
    return JSONResponse(status_code=400, content={"message": "cannot decode input"})


if __name__ == "__main__":
    api = ls.examples.SimpleExceptionAPI(InvalidInputError)
    server = ls.LitServer(api, exc_handlers=[(InvalidInputError, invalid_input_error_handler)])
    server.run(port=8000)
