package com.khanhlms.medical_store.dtos.cartItems.request;

import lombok.*;
import lombok.experimental.FieldDefaults;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CreateCartItemRequest {
    String productId;
    Integer quantity;
}
