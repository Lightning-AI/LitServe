from fastmcp import Client


async def test_mcp_server():
    async with Client("http://localhost:8000/mcp/sse") as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        assert len(tools) == 1


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_mcp_server())
