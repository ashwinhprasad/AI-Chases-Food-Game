# importing the libraries
import pygame, math, random, torch

# game configurations
WIDTH,HEIGHT = 900,600
WIN = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Gradient Descent")
FPS = 60
GREEK_IMAGE = pygame.transform.scale(pygame.image.load("./Assets/greek.png"),(90,60))
ATHENS_IMAGE = pygame.transform.scale(pygame.image.load("./Assets/athens.png"),(80,80))
FOOD_IMAGE = pygame.transform.scale(pygame.image.load("./Assets/food.png"),(50,50))
ATHENS_VEL = 2
GREEK_VEL = 8
GREEK_SCORE = 0
FINAL_SCORE = 10
ATHENS_SCORE = 0


# window and player settings
def drawWindow(athens,greek,food):
    WIN.fill((255,255,255))
    WIN.blit(GREEK_IMAGE,(greek.x,greek.y))
    WIN.blit(ATHENS_IMAGE,(athens.x,athens.y))
    WIN.blit(FOOD_IMAGE,(food.x,food.y))
    pygame.display.update()

# handle player movement
def handlePlayerMovement(greek,keys_pressed):
    if keys_pressed[pygame.K_a]: # left
        greek.x -= GREEK_VEL
    if keys_pressed[pygame.K_d]: # right
        greek.x += GREEK_VEL
    if keys_pressed[pygame.K_w]: # up
        greek.y -= GREEK_VEL
    if keys_pressed[pygame.K_s]: # down
        greek.y += GREEK_VEL

# when to quit the game
def endgame(athens,greek,food):
    if( math.sqrt(pow((athens.x - food.x),2) + pow((athens.y - food.y),2)) <= 45):
        food.x = random.randint(50,850)
        food.y = random.randint(50,550)
        global ATHENS_SCORE
        ATHENS_SCORE += 1

    if(math.sqrt(pow((greek.x - food.x),2) + pow((greek.y - food.y),2)) <= 45):
        food.x = random.randint(50,850)
        food.y = random.randint(50,550)
        global GREEK_SCORE
        GREEK_SCORE += 1
    
    if ATHENS_SCORE == FINAL_SCORE:
        return (False,"AI")
    elif GREEK_SCORE == FINAL_SCORE:
        return (False,"Player")
    else:
        return (True,0)


# AI logic
def AImovement(athens,food):
    
    # food position
    food_x = torch.tensor(food.x)
    food_y = torch.tensor(food.y)

    for i in range(2):

        # hero position
        athens_x = torch.tensor(float(athens.x),requires_grad=True)
        athens_y = torch.tensor(float(athens.y),requires_grad=True)

        # calculate loss
        loss = torch.sqrt((athens_x - food_x)**2 + (athens_y - food_y)**2)

        # if i%100==0:
        #    print(loss)

        # gradient descent
        loss.backward()

        with torch.no_grad():
            # print(hero_x.grad,hero_y.grad)

            if athens_x.grad < 0:
                athens_x += ATHENS_VEL
            else:
                athens_x -= ATHENS_VEL

            if athens_y.grad < 0:
                athens_y += ATHENS_VEL  
            else:
                athens_y -= ATHENS_VEL             
        
        if loss >= 40:
            athens.x = int(athens_x.item())
            athens.y = int(athens_y.item())

        # reset gradients
        athens_x.grad.zero_()
        athens_y.grad.zero_()

# main game loop
def main():
    athens = pygame.Rect(100,200,50,50)
    greek = pygame.Rect(200,300,50,50)
    food = pygame.Rect(200,300,50,50)
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        keys_pressed = pygame.key.get_pressed()
        handlePlayerMovement(greek,keys_pressed) 
        drawWindow(athens,greek,food)
        AImovement(athens,food)
        (run,winner) = endgame(athens,greek,food)
    print(f"{winner} won")
    pygame.quit()

if __name__ == "__main__":
    main()