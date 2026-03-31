-- Создаётся автоматически при первом запуске контейнера
-- Скрипты в /docker-entrypoint-initdb.d/ выполняются только один раз!

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO users (username, email) VALUES 
    ('admin', 'admin@example.com'),
    ('test_user', 'test@example.com');

-- Создание дополнительного пользователя с ограниченными правами
CREATE USER app_user WITH PASSWORD 'app_password';
GRANT SELECT, INSERT, UPDATE ON users TO app_user;
GRANT USAGE, SELECT ON SEQUENCE users_id_seq TO app_user;