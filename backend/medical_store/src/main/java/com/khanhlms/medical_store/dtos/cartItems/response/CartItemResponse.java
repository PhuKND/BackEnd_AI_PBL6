package com.khanhlms.medical_store.dtos.cartItems.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

import java.time.LocalDateTime;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
public class CartItemResponse {
    String productId;
    String productName;
    String imageUrl;
    Integer quantity;
    Double totalPrice;
    String currency;
    Double originPrice;
    String messageDiscount;
    LocalDateTime discountStart;
    LocalDateTime discountEnd;
    Double percent;
    String unit;
}
