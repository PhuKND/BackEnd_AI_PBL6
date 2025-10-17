package com.khanhlms.medical_store.services;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.khanhlms.medical_store.dtos.products.requests.CreateProductRequest;
import com.khanhlms.medical_store.dtos.products.requests.IngredientRequest;
import com.khanhlms.medical_store.dtos.products.response.CreateProductResponse;
import com.khanhlms.medical_store.dtos.products.response.DetailProduct;
import com.khanhlms.medical_store.dtos.products.response.ProductResponse;
import com.khanhlms.medical_store.entities.IngredientEntity;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.mapper.ProductsMapper;
import com.khanhlms.medical_store.repositories.ProductRepository;
import com.khanhlms.medical_store.utills.BaseRedisUtils;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class ProductsSercvice {
    ProductRepository productRepository;
    ProductsMapper  productsMapper;
    BaseRedisUtils redisUtils;
    private ObjectMapper objectMapper = new  ObjectMapper();
    public CreateProductResponse createProduct(CreateProductRequest request, List<IngredientRequest>  ingredients) {
        ProductsEntity productsEntity = this.productsMapper.toEntity(request);
        Date productDate = productsEntity.getProductDate();
        Date expirationDate = productsEntity.getExpirationDate();
        if (productDate.getTime() > expirationDate.getTime()) {
            throw new AppException(ErrorCode.EXPIRERATION_EXCEPTION);
        }
        productsEntity.setIngredients(mapIngredientEntity(ingredients));
        productsEntity.setIsActive(true);
        productsEntity.setIsDeleted(false);
        productsEntity.setRatingAvg(0.0);
        productsEntity.setSoldQuantity(0);
        productsEntity.setPosition((int)this.productRepository.count());
        if (productsEntity.getImages() != null) {
            productsEntity.getImages().forEach(img -> img.setProduct(productsEntity));
        }
        if (productsEntity.getIngredients() != null) {
            productsEntity.getIngredients().forEach(ingredient -> ingredient.setProduct(productsEntity));
        }

        return this.productsMapper.toCreateProductResponse(productRepository.save(productsEntity));
    }

    private List<IngredientEntity> mapIngredientEntity(List<IngredientRequest> ingredientRequests){
        if (ingredientRequests == null || ingredientRequests.isEmpty()) {return Collections.emptyList();}
        return ingredientRequests.stream()
                .map(ingredientRequest -> {
                    return  IngredientEntity.builder()
                            .name(ingredientRequest.getName())
                            .description(ingredientRequest.getDescription())
                            .amount(ingredientRequest.getAmount())
                            .unit(ingredientRequest.getUnit())
                            .build();
                })
                .toList();
    }

    public List<ProductResponse> handGetProduct( String redisKey,Pageable pageable) {
        List<ProductResponse> result = null;
        if(Objects.isNull(this.redisUtils.getForString(redisKey))) {
             result = this.productRepository.findAllByIsActiveTrueAndIsDeletedFalse(pageable)
                    .stream()
                    .map(item -> this.productsMapper.toProductResponse(item))
                    .toList();
            try {
                String valueString = this.objectMapper.writeValueAsString(result);
                this.redisUtils.set(redisKey,valueString, 1l,  TimeUnit.SECONDS);
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        }
        else {
            String dataRedis = (String) this.redisUtils.getForString(redisKey);
            try {
                // convert JSON string to List<ProductResponse>
                result = objectMapper.readValue(
                        dataRedis,
                        objectMapper.getTypeFactory().constructCollectionType(List.class, ProductResponse.class)
                );
            } catch (JsonProcessingException e) {
                throw new RuntimeException("Error parsing product list from Redis", e);
            }
        }
        return result;
    }
    public DetailProduct handGetDetailProduct(String productId) {
        ProductsEntity product = this.productRepository.findById(productId)
                .orElseThrow(() -> new AppException(ErrorCode.PRODUCT_NOT_FOUND));
        return productsMapper.toDetailProduct(product);
    }
    public List<ProductResponse> getByKeyword(String keyword, Pageable pageable) {
        if (keyword.equals("")) return Collections.emptyList();
        return this.productRepository.getProductsByKeyword(keyword).stream()
                .map(product -> this.productsMapper.toProductResponse(product))
                .toList();
    }
}
