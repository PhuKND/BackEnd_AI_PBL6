package com.khanhlms.medical_store.repositories.custom.impl;

import com.khanhlms.medical_store.dtos.products.response.ProductResponse;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.repositories.custom.ProductCustom;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

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
}
