package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.dtos.frequently.request.CreateFrequentlyRequest;
import com.khanhlms.medical_store.dtos.frequently.response.FrequentlyResponse;
import com.khanhlms.medical_store.entities.FrequentlyEntity;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.mapper.FrequentlyMapper;
import com.khanhlms.medical_store.repositories.FrequentlyRepository;
import com.khanhlms.medical_store.repositories.ProductRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.stereotype.Service;

import java.time.LocalDate;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class FrequentlyService {
    FrequentlyRepository frequentlyRepository;
    ProductRepository productRepository;
    FrequentlyMapper frequentlyMapper;

    public FrequentlyResponse handCrateFrequently(String productID, CreateFrequentlyRequest createFrequentlyRequest) {
        ProductsEntity  product = this.productRepository.findById(productID)
                .orElseThrow(() -> new AppException(ErrorCode.PRODUCT_NOT_FOUND));
        FrequentlyEntity frequently = this.frequentlyMapper.toEntity(createFrequentlyRequest);
        frequently.setProduct(product);
        frequently.setCreatedDate(LocalDate.now());
        return this.frequentlyMapper.toResponse(this.frequentlyRepository.save(frequently));
    }
}
