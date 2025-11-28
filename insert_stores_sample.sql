-- STORES 테이블 샘플 데이터 INSERT
-- 사용 전에 users와 image_assets 테이블에 데이터가 있는지 확인하세요.

-- 샘플 1: 부대찌개 전문점 (첫 번째 유저)
INSERT INTO stores (
    user_id,
    image_id,
    title,
    body,
    store_category,
    auto_scoring_flag,
    uid
) VALUES (
    (SELECT user_id FROM users ORDER BY pk LIMIT 1 OFFSET 0),
    (SELECT image_asset_id FROM image_assets LIMIT 1),
    '맛있는 부대찌개 전문점',
    '진한 국물과 든든한 재료가 일품인 부대찌개를 맛볼 수 있는 전문점입니다. 신선한 햄과 소시지, 라면사리가 듬뿍 들어간 정통 부대찌개를 제공합니다.',
    'restaurant',
    FALSE,
    'store-budae-jjigae-001'
);

-- 샘플 2: 김치찌개 전문점 (두 번째 유저)
INSERT INTO stores (
    user_id,
    image_id,
    title,
    body,
    store_category,
    auto_scoring_flag,
    uid
) VALUES (
    (SELECT user_id FROM users ORDER BY pk LIMIT 1 OFFSET 1),
    (SELECT image_asset_id FROM image_assets LIMIT 1),
    '시원한 김치찌개 맛집',
    '깊고 진한 맛의 김치찌개를 선사하는 전문점입니다. 잘 익은 김치와 돼지고기가 조화롭게 어우러진 정통 김치찌개와 함께 밑반찬까지 푸짐하게 제공합니다.',
    'restaurant',
    TRUE,
    'store-kimchi-jjigae-002'
);

-- 샘플 3: 파스타 전문점 (첫 번째 유저)
INSERT INTO stores (user_id, title, body, store_category, uid) VALUES ((SELECT user_id FROM users LIMIT 1), '새로운 파스타 가게', '알리오 올리오 전문점입니다.', 'restaurant', 'store-pasta-003');

