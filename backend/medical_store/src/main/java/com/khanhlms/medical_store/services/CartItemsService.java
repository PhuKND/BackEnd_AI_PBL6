package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.dtos.cartItems.request.CreateCartItemRequest;
import com.khanhlms.medical_store.dtos.cartItems.response.CartItemResponse;
import com.khanhlms.medical_store.dtos.cartItems.response.CreateCartItemResponse;
import com.khanhlms.medical_store.entities.CartItemEntity;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.entities.UserEntity;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.mapper.CartsMapper;
import com.khanhlms.medical_store.repositories.CartItemsRespository;
import com.khanhlms.medical_store.repositories.ProductRepository;
import com.khanhlms.medical_store.repositories.UserRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE,  makeFinal = true)
public class CartItemsService {
    CartItemsRespository cartItemsRespository;
    UserRepository userRepository;
    ProductRepository productRepository;
    CartsMapper  cartsMapper;
    public CreateCartItemResponse handCreateCartItems( String username,CreateCartItemRequest createCartItemRequest) {
        CartItemEntity cartItem = null;
        UserEntity user = userRepository.findByUsername(username)
                .orElseThrow(() -> new AppException(ErrorCode.USER_NOT_EXISTED));
        ProductsEntity product = this.productRepository.findById(createCartItemRequest.getProductId())
                .orElseThrow(() -> new AppException(ErrorCode.PRODUCT_NOT_FOUND));
        int quantity = createCartItemRequest.getQuantity();
        if (quantity <= 0) {
            throw new  AppException(ErrorCode.INVALID_REQUEST);
        }
        if (this.cartItemsRespository.findByProductAndUser(product, user).isPresent()){
            cartItem = this.cartItemsRespository.findByProductAndUser(product, user).get();
            cartItem.setQuantity(cartItem.getQuantity() + quantity);
        }
        else {
            cartItem = CartItemEntity.builder()
                    .user(user)
                    .product(product)
                    .quantity(createCartItemRequest.getQuantity())
                    .deleted(false)
                    .build();
        }
        cartItem = this.cartItemsRespository.save(cartItem);
        return CreateCartItemResponse.builder()
                .cartItemId(cartItem.getId())
                .userId(user.getId())
                .productId(product.getId())
                .quantity(cartItem.getQuantity())
                .build();
    }
    public List<CartItemResponse> handMyCartItems(String username, Pageable  pageable) {
        return  this.cartItemsRespository.findAllByDeletedFalse(pageable)
                .get()
                .map(item -> this.cartsMapper.toCartItemResponse(item))
                .toList();
    }

}
