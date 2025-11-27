package com.khanhlms.medical_store.repositories.custom.impl;

import com.khanhlms.medical_store.dtos.products.response.ProductResponse;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.repositories.custom.ProductCustom;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;

@Repository
public class ProductCustomImpl implements ProductCustom {
    @PersistenceContext
    private EntityManager em;

    @Override
    public List<ProductsEntity> getProductsByKeyword(String keyword) {
        String sql = """
            SELECT * FROM products
            WHERE LOWER(name) LIKE LOWER(CONCAT('%', :keyword, '%'))
               OR LOWER(benefit) LIKE LOWER(CONCAT('%', :keyword, '%'))
               AND is_active = 1
               AND is_deleted = 0
            ORDER BY position ASC
            """;

        Query query = em.createNativeQuery(sql, ProductsEntity.class);
        query.setParameter("keyword", keyword);

        return query.getResultList();
    }

    @Override
    public List<ProductsEntity> filterProducts(Map<String, String> filters, Integer page, Integer size) {
        StringBuilder sql = new StringBuilder("""
        SELECT p.* FROM products p
        JOIN manufacturers m ON p.manufacturer_id = m.id
        WHERE p.is_active = 1
          AND p.is_deleted = 0
    """);

        // ==========================
        // ⚡ FILTER CONDITIONS
        // ==========================
        if (filters.containsKey("categoryId")) {
            sql.append(" AND p.category_id = :categoryId");
        }

        if (filters.containsKey("manufacturerId")) {
            sql.append(" AND p.manufacturer_id = :manufacturerId");
        }

        if (filters.containsKey("origin")) {
            sql.append(" AND LOWER(m.country) LIKE LOWER(CONCAT('%', :origin, '%'))");
        }

        if (filters.containsKey("minPrice")) {
            sql.append(" AND p.origin_price >= :minPrice");
        }

        if (filters.containsKey("maxPrice")) {
            sql.append(" AND p.origin_price <= :maxPrice");
        }

        // ==========================
        // 🔥 SORT
        // ==========================
        String sort = filters.get("sort");
        if ("price_asc".equals(sort)) {
            sql.append(" ORDER BY p.origin_price ASC");
        } else if ("price_desc".equals(sort)) {
            sql.append(" ORDER BY p.origin_price DESC");
        } else {
            sql.append(" ORDER BY p.position ASC"); // default sorting
        }

        // ==========================
        // 📌 PAGINATION (page start = 0)
        // ==========================
        if (page == null || page < 0) page = 0;
        if (size == null || size <= 0) size = 10;

        int offset = page * size;
        sql.append(" LIMIT :size OFFSET :offset");

        Query query = em.createNativeQuery(sql.toString(), ProductsEntity.class);

        // ==========================
        // 🔥 SET PARAMS
        // ==========================
        filters.forEach((key, value) -> {
            switch (key) {
                case "categoryId" -> query.setParameter("categoryId", value);
                case "manufacturerId" -> query.setParameter("manufacturerId", value);
                case "origin" -> query.setParameter("origin", value);
                case "minPrice" -> query.setParameter("minPrice", Double.parseDouble(value));
                case "maxPrice" -> query.setParameter("maxPrice", Double.parseDouble(value));
            }
        });

        // SET PAGING PARAMS
        query.setParameter("size", size);
        query.setParameter("offset", offset);

        return query.getResultList();
    }
}
