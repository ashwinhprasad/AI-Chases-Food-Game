import pygame, math, random, torch


WIDTH,HEIGHT = 900,600
WIN = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Gradient Descent")
FPS = 60
HERO_IMAGE = pygame.transform.scale(pygame.image.load("./Assets/hero.png"),(50,50))
FOOD_IMAGE = pygame.transform.scale(pygame.image.load("./Assets/food.png"),(50,50))
VEL = 3
BACKGROUND = pygame.transform.scale(pygame.image.load('./Assets/background.jpeg'),(900,600))

def drawWindow(hero,food):
    WIN.fill((255,255,255))
    WIN.blit(HERO_IMAGE,(hero.x,hero.y))
    WIN.blit(FOOD_IMAGE,(food.x,food.y))
    pygame.display.update()

def handleHeroMovement(hero,keys_pressed):
    if keys_pressed[pygame.K_a]: # left
        hero.x -= VEL
    if keys_pressed[pygame.K_d]: # right
        hero.x += VEL
    if keys_pressed[pygame.K_w]: # up
        hero.y -= VEL
    if keys_pressed[pygame.K_s]: # down
        hero.y += VEL

def AImovement(hero,food):
    
    # food position
    food_x = torch.tensor(food.x)
    food_y = torch.tensor(food.y)

    # learing rate
    alpha = 0.1

    for i in range(2):

        # hero position
        hero_x = torch.tensor(float(hero.x),requires_grad=True)
        hero_y = torch.tensor(float(hero.y),requires_grad=True)

        # calculate loss
        loss = torch.sqrt((hero_x - food_x)**2 + (hero_y - food_y)**2)

        # if i%100==0:
        #    print(loss)

        # gradient descent
        loss.backward()

        with torch.no_grad():
            # print(hero_x.grad,hero_y.grad)

            if hero_x.grad < 0:
                hero_x += VEL
            else:
                hero_x -= VEL

            if hero_y.grad < 0:
                hero_y += VEL  
            else:
                hero_y -= VEL             
        
        if loss >= 40:
            hero.x = int(hero_x.item())
            hero.y = int(hero_y.item())

        # reset gradients
        hero_x.grad.zero_()
        hero_y.grad.zero_()

def main():
    hero = pygame.Rect(100,200,50,50)
    food = pygame.Rect(200,300,50,50)
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        keys_pressed = pygame.key.get_pressed()
        # handleHeroMovement(hero,keys_pressed)
        drawWindow(hero,food)
        AImovement(hero,food)
        if( math.sqrt(pow((hero.x - food.x),2) + pow((hero.y - food.y),2)) <= 45 ):
            food.x = random.randint(50,850)
            food.y = random.randint(50,550)

    pygame.quit()

if __name__ == "__main__":
    main()