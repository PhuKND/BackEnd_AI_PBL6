package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.dtos.discounts.request.CreateDiscountRequest;
import com.khanhlms.medical_store.dtos.discounts.response.CreateDiscountResponse;
import com.khanhlms.medical_store.entities.DiscountEntity;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.repositories.DiscountRepository;
import com.khanhlms.medical_store.repositories.ProductRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class DiscountService {
    DiscountRepository discountRepository;
    ProductRepository productRepository;
    public CreateDiscountResponse handCreateDiscount( String productID, CreateDiscountRequest request) {
        if (request.getPercent() <= 0){
            throw new AppException(ErrorCode.INVALID_REQUEST);
        }
        if (request.getStartDate().isAfter( request.getEndDate() )){
            throw new AppException(ErrorCode.INVALID_REQUEST);
        }
        ProductsEntity products = this.productRepository.findById(productID)
                .orElseThrow(() ->new AppException(ErrorCode.PRODUCT_NOT_FOUND));
        DiscountEntity  discount = null;
        if (this.discountRepository.findByProduct(products).isPresent()){
            discount = this.discountRepository.findByProduct(products).get();
            discount.setPercent(request.getPercent());
            discount.setStartDate(request.getStartDate());
            discount.setEndDate(request.getEndDate());
            discount.setMessage(request.getMessage());
        }
        else {
            discount = DiscountEntity.builder()
                    .percent(request.getPercent())
                    .message(request.getMessage())
                    .startDate(request.getStartDate())
                    .endDate(request.getEndDate())
                    .product(products)
                    .build();
        }
        discount = this.discountRepository.save(discount);
        return CreateDiscountResponse.builder()
                .productId(discount.getProduct().getId())
                .percent(discount.getPercent())
                .message(discount.getMessage())
                .startDate(discount.getStartDate())
                .endDate(discount.getEndDate())
                .build();
    }
}
