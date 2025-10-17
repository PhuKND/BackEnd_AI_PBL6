package com.khanhlms.medical_store.repositories.custom;

import com.khanhlms.medical_store.dtos.products.response.ProductResponse;
import com.khanhlms.medical_store.entities.ProductsEntity;

import java.util.List;

public interface ProductCustom {
    List<ProductsEntity> getProductsByKeyword(String keyword);
}
