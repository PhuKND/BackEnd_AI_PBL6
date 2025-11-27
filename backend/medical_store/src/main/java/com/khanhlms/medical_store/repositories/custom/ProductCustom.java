package com.khanhlms.medical_store.repositories.custom;

import com.khanhlms.medical_store.dtos.products.response.ProductResponse;
import com.khanhlms.medical_store.entities.ProductsEntity;

import java.util.List;
import java.util.Map;

public interface ProductCustom {
    List<ProductsEntity> getProductsByKeyword(String keyword);
    List<ProductsEntity> filterProducts(Map<String, String> filters,Integer page, Integer size);
}
