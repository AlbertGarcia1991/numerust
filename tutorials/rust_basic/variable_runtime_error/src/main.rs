use std::io;

fn main() {
    let a: [u8; 5] = [1, 2, 3, 4, 5];

    println!("Please enter an array index.");

    let mut index = String::new();

    io::stdin()
        .read_line(&mut index)
        .expect("Failed to read line");

    let index: usize = index
        .trim()
        .parse()
        .expect("Index entered was not a number");

    let element = a[index] - 2;

    println!("The value of the element at index {index} is: {element}");
}
