docker run --name my-postgres16 \
-e POSTGRES_PASSWORD=mysecretpassword \
-e POSTGRES_USER=user1 \
-p 5432:5432 \
-v pgdata:/var/lib/postgresql/data \
-d postgres:16