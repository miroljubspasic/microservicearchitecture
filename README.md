cp .env.example .env
cp .env ingest/.env
cp frontend/.env.example frontend/.env
cp service2/.env.example service2/.env
cp .env service3/.env




docker compose up


to run ingest:

docker compose run ingest



docker compose stop


docker compose down --rmi all